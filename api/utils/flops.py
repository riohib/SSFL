import logging
import numpy as np
import os

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models



def print_model_param_nums(model=None):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([(param!=0).sum() if len(param.size()) == 4 or len(param.size()) == 2 else 0 for name,param in model.named_parameters()])
    print('  + Number of params: %.2f' % (total))

def count_training_flops(model,dataset, full=False):
    flops = 3* count_model_param_flops(model,dataset, full=full)
    return flops

def count_inference_flops (model,dataset):
    flops = count_model_param_flops(model,dataset)
    return flops

def count_model_param_flops(model=None,dataset=None, multiply_adds=True, full=False):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0
        if not full:
            num_weight_params = (self.weight.data != 0).float().sum()
        else:
            num_weight_params= torch.numel(self.weight.data)
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size
        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        if not full:
            weight_ops = (self.weight.data != 0).float().sum() * (2 if multiply_adds else 1)
            bias_ops = (self.bias.data != 0).float().sum() if self.bias is not None else 0
        else:
            weight_ops = torch.numel(self.weight.data) * (2 if multiply_adds else 1)
            bias_ops = torch.numel(self.bias.data) if self.bias is not None else 0
        flops = batch_size * (weight_ops + bias_ops)
        # logging.info("L{}".format(flops))
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(handles,net):

        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                handles+=[net.register_forward_hook(conv_hook)]
            if isinstance(net, torch.nn.Linear):
                handles+= [net.register_forward_hook(linear_hook)]
            return
        for c in childrens:
            foo(handles,c)

    handles=[]
    foo(handles,model)
    if dataset.name == 'cifar100':
        input_res = 32
        input_channel = 3
    elif dataset.name == 'cifar10':
        input_res = 32
        input_channel = 3
    elif dataset.name == 'tiny':
        input_res = 64
        input_channel = 3
    elif dataset.name == 'imagenet':
        input_res = 224
        input_channel = 3
    elif dataset.name == 'cub':
        input_res = 224
        input_channel = 3
    else:
        # Default case or error
        raise ValueError(f"FLOPs counter does not recognize dataset: {dataset.name}")
    
    device = next(model.parameters()).device
    input = Variable(torch.rand(input_channel,input_res,input_res).unsqueeze(0), requires_grad = True).to(device)

    model.eval()

    out = model(input)

    model.train()
    
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    for handle in handles:
        handle.remove()
    # print('  + Number of FLOPs: %.2f' % (total_flops))
    return total_flops
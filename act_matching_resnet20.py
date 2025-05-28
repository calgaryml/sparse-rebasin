"""
Code taken from REPAIR: REnormalizing Permuted Activations for Interpolation Repair.
https://github.com/KellerJordan/REPAIR
Authors: Keller Jordan and Hanie Sedghi and Olga Saukh and Rahim Entezari and Behnam Neyshabur

We modified the code base to permute sparse models (i.e. binary weight masks).

"""

from tqdm import tqdm
import numpy as np
import scipy.optimize

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torchvision
import torchvision.transforms as T
from models import ResNet
import yaml


def run_corr_matrix(net0, net1, loader, device="cuda"):
    """
    Code taken from REPAIR notebook
    Args: net0 -> torch model
          net1 -> torch model which needs to be permuted
    """
    n = len(loader)
    mean0 = mean1 = std0 = std1 = None
    with torch.no_grad():
        net0.eval()
        net1.eval()
        for i, (images, _) in enumerate(tqdm(loader)):

            img_t = images.float().cuda()
            out0 = net0(img_t)
            out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
            out0 = out0.reshape(-1, out0.shape[2]).double()

            out1 = net1(img_t)
            out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
            out1 = out1.reshape(-1, out1.shape[2]).double()

            mean0_b = out0.mean(dim=0)
            mean1_b = out1.mean(dim=0)
            std0_b = out0.std(dim=0)
            std1_b = out1.std(dim=0)
            outer_b = (out0.T @ out1) / out0.shape[0]

            if i == 0:
                mean0 = torch.zeros_like(mean0_b)
                mean1 = torch.zeros_like(mean1_b)
                std0 = torch.zeros_like(std0_b)
                std1 = torch.zeros_like(std1_b)
                outer = torch.zeros_like(outer_b)
            mean0 += mean0_b / n
            mean1 += mean1_b / n
            std0 += std0_b / n
            std1 += std1_b / n
            outer += outer_b / n

    cov = outer - torch.outer(mean0, mean1)
    corr = cov / (torch.outer(std0, std1) + 1e-4)
    return corr

def get_layer_perm1(corr_mtx):
    corr_mtx_a = corr_mtx.cpu().numpy()
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)
    assert (row_ind == np.arange(len(corr_mtx_a))).all()
    perm_map = torch.tensor(col_ind).long()
    return perm_map

# returns the channel-permutation to make layer1's activations most closely
# match layer0's.
def get_layer_perm(net0, net1, loader):
    corr_mtx = run_corr_matrix(net0, net1, loader)
    return get_layer_perm1(corr_mtx)

# modifies the weight matrices of a convolution and batchnorm
# layer given a permutation of the output channels
def permute_output(perm_map, conv, bn=None):
    pre_weights = [conv.weight]
    if conv.bias is not None:
        pre_weights.append(conv.bias)
    if bn is not None:
        pre_weights.extend([bn.weight, bn.bias, bn.running_mean, bn.running_var])
    
    if hasattr(conv, 'weight_orig'):
        pre_weights.append(conv.weight_orig)
    if hasattr(conv, 'weight_mask'):
        pre_weights.append(conv.weight_mask)

    for w in pre_weights:
        w.data = w[perm_map]

# modifies the weight matrix of a convolution layer for a given
# permutation of the input channels
def permute_input(perm_map, conv):
    w = conv.weight
    w.data = w[:, perm_map, :, :]

    if hasattr(conv, 'weight_orig'):
        conv.weight_orig.data = conv.weight_orig[:, perm_map, :, :]

    if hasattr(conv, 'weight_mask'):
        conv.weight_mask.data = conv.weight_mask[:, perm_map, :, :]

def get_blocks(net):
    return nn.Sequential(nn.Sequential(net.conv1, net.bn1, nn.ReLU()),
                         *net.layer1, *net.layer2, *net.layer3)

def permute_model_resnet(model0, model1, modelA_sparse, loader, config):
    """
    Permuting model0, by matching the activations of model0 to model1 and then applying perm to model0 and modelA_sparse.
    So, when interpolating between pi(model0) and model1 you should expect a small error barrier.
    """
    blocks0 = get_blocks(model0)
    blocks1 = get_blocks(model1)
    blocks_sparse = get_blocks(modelA_sparse)

    for k in range(1, len(blocks0)):
        block0 = blocks0[k]
        block1 = blocks1[k]
        block_sparse = blocks_sparse[k]
        subnet0 = nn.Sequential(blocks0[:k], block0.conv1, block0.bn1, nn.ReLU())
        subnet1 = nn.Sequential(blocks1[:k], block1.conv1, block1.bn1, nn.ReLU())
        perm_map = get_layer_perm(subnet1, subnet0, loader)
        permute_output(perm_map, block0.conv1, block0.bn1)
        permute_input(perm_map, block0.conv2)
        permute_output(perm_map, block_sparse.conv1, block_sparse.bn1)
        permute_input(perm_map, block_sparse.conv2)

    kk = [3, 6, 8]

    perm_map = get_layer_perm(blocks1[:kk[0]+1], blocks0[:kk[0]+1], loader)
    permute_output(perm_map, model0.conv1, model0.bn1)
    permute_output(perm_map, modelA_sparse.conv1, modelA_sparse.bn1)
    for block in model0.layer1:
        permute_input(perm_map, block.conv1)
        permute_output(perm_map, block.conv2, block.bn2)
    for block in modelA_sparse.layer1:
        permute_input(perm_map, block.conv1)
        permute_output(perm_map, block.conv2, block.bn2)
    block = model0.layer2[0]
    permute_input(perm_map, block.conv1)
    permute_input(perm_map, block.shortcut[0])
    block_sparse = modelA_sparse.layer2[0]
    permute_input(perm_map, block_sparse.conv1)
    permute_input(perm_map, block_sparse.shortcut[0])

    perm_map = get_layer_perm(blocks1[:kk[1]+1], blocks0[:kk[1]+1], loader)
    for i, block in enumerate(model0.layer2):
        if i > 0:
            permute_input(perm_map, block.conv1)
        else:
            permute_output(perm_map, block.shortcut[0], block.shortcut[1])
        permute_output(perm_map, block.conv2, block.bn2)
    for i, block in enumerate(modelA_sparse.layer2):
        if i > 0:
            permute_input(perm_map, block.conv1)
        else:
            permute_output(perm_map, block.shortcut[0], block.shortcut[1])
        permute_output(perm_map, block.conv2, block.bn2)
    block = model0.layer3[0]
    permute_input(perm_map, block.conv1)
    permute_input(perm_map, block.shortcut[0])
    block_sparse = modelA_sparse.layer3[0]
    permute_input(perm_map, block_sparse.conv1)
    permute_input(perm_map, block_sparse.shortcut[0])

    perm_map = get_layer_perm(blocks1[:kk[2]+1], blocks0[:kk[2]+1], loader)
    for i, block in enumerate(model0.layer3):
        if i > 0:
            permute_input(perm_map, block.conv1)
        else:
            permute_output(perm_map, block.shortcut[0], block.shortcut[1])
        permute_output(perm_map, block.conv2, block.bn2)
    for i, block in enumerate(modelA_sparse.layer3):
        if i > 0:
            permute_input(perm_map, block.conv1)
        else:
            permute_output(perm_map, block.shortcut[0], block.shortcut[1])
        permute_output(perm_map, block.conv2, block.bn2)
    model0.linear.weight.data = model0.linear.weight[:, perm_map]
    modelA_sparse.linear.weight.data = modelA_sparse.linear.weight[:, perm_map]
    if hasattr(modelA_sparse.linear, 'weight_orig'):
        
        modelA_sparse.linear.weight_orig.data = modelA_sparse.linear.weight_orig[:, perm_map]

    if hasattr(modelA_sparse.linear, 'weight_mask'):
        
        modelA_sparse.linear.weight_mask.data = modelA_sparse.linear.weight_mask[:, perm_map]
    
    return model0, modelA_sparse
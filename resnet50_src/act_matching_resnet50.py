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
import torchvision.transforms as T
from models import get_blocks


def run_corr_matrix(net0, net1, loader, device="cuda"):
    """
    Given two networks net0, net1 which each output a feature map of shape NxCxWxH, this will reshape both outputs to (N*W*H)xC 
    and then compute a CxC correlation matrix between the two.
    """
    n = len(loader)
    with torch.no_grad():
        net0.eval()
        net1.eval()
        for i, (images, _) in enumerate(tqdm(loader)):
            
            img_t = images.float().cuda()
            out0 = net0(img_t).double()
            out0 = out0.permute(0, 2, 3, 1).reshape(-1, out0.shape[1])
            out1 = net1(img_t).double()
            out1 = out1.permute(0, 2, 3, 1).reshape(-1, out1.shape[1])

            # save batchwise first+second moments and outer product
            mean0_b = out0.mean(dim=0)
            mean1_b = out1.mean(dim=0)
            sqmean0_b = out0.square().mean(dim=0)
            sqmean1_b = out1.square().mean(dim=0)
            outer_b = (out0.T @ out1) / out0.shape[0]
            if i == 0:
                mean0 = torch.zeros_like(mean0_b)
                mean1 = torch.zeros_like(mean1_b)
                sqmean0 = torch.zeros_like(sqmean0_b)
                sqmean1 = torch.zeros_like(sqmean1_b)
                outer = torch.zeros_like(outer_b)
            mean0 += mean0_b / n
            mean1 += mean1_b / n
            sqmean0 += sqmean0_b / n
            sqmean1 += sqmean1_b / n
            outer += outer_b / n

    cov = outer - torch.outer(mean0, mean1)
    std0 = (sqmean0 - mean0**2).sqrt()
    std1 = (sqmean1 - mean1**2).sqrt()
    corr = cov / (torch.outer(std0, std1) + 1e-4)
    return corr

def get_layer_perm1(corr_mtx):
    corr_mtx_a = corr_mtx.cpu().numpy()
    corr_mtx_a = np.nan_to_num(corr_mtx_a)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)
    assert (row_ind == np.arange(len(corr_mtx_a))).all()
    perm_map = torch.tensor(col_ind).long()
    return perm_map

# returns the channel-permutation to make layer1's activations most closely
# match layer0's. --> so this is permuting model1  --> model0 (i.e. π(net1))
def get_layer_perm(net0, net1, loader):
    corr_mtx = run_corr_matrix(net0, net1, loader)
    return get_layer_perm1(corr_mtx)

def permute_output(perm_map, conv, bn=None):
    pre_weights = [conv.weight]
    if bn is not None:
        pre_weights.extend([bn.weight, bn.bias, bn.running_mean, bn.running_var])
        
    if hasattr(conv, 'weight_orig'):
        pre_weights.append(conv.weight_orig)
    if hasattr(conv, 'weight_mask'):
        pre_weights.append(conv.weight_mask)
        
    for w in pre_weights:
        w.data = w[perm_map]

def permute_input(perm_map, layer):
    w = layer.weight
    w.data = w[:, perm_map]
    
    if hasattr(layer, 'weight_orig'):
        layer.weight_orig.data = layer.weight_orig[:, perm_map]
    if hasattr(layer, 'weight_mask'):
        layer.weight_mask.data = layer.weight_mask[:, perm_map]
        
# Restrict the permutations such that the network is not functionally changed.
# In particular, the same permutation must be applied to every conv output in a residual stream.
def get_permk(k):
    if k == 0:
        return 0
    elif k > 0 and k <= 3:
        return 3
    elif k > 3 and k <= 7:
        return 7
    elif k > 7 and k <= 13:
        return 13
    elif k > 13 and k <= 16:
        return 16
    else:
        raise Exception()
    
def permute_model_resnet50(model0, model1, modelA_sparse, loader, config):
    last_kk = None
    blocks0 = get_blocks(model0)
    blocks1 = get_blocks(model1)
    blocks_sparse = get_blocks(modelA_sparse)
    
    for k in range(1, len(blocks0)):
        block0 = blocks0[k]
        block1 = blocks1[k]
        block_sparse = blocks_sparse[k]
        subnet0 = nn.Sequential(blocks0[:k],
                                block0.conv1, block0.bn1, block0.relu)
        subnet1 = nn.Sequential(blocks1[:k],
                                block1.conv1, block1.bn1, block1.relu)
        # perm_map = get_layer_perm(subnet0, subnet1, train_dl) --> original if you want to permute model1 to model0
        perm_map = get_layer_perm(subnet1, subnet0, loader)
        permute_output(perm_map, block0.conv1, block0.bn1)
        permute_input(perm_map, block0.conv2)
        permute_output(perm_map, block_sparse.conv1,block_sparse.bn1)
        permute_input(perm_map, block_sparse.conv2)
        
        subnet0 = nn.Sequential(blocks0[:k],
                                block0.conv1, block0.bn1, block0.relu,
                                block0.conv2, block0.bn2, block0.relu)
        subnet1 = nn.Sequential(blocks1[:k],
                                block1.conv1, block1.bn1, block1.relu,
                                block1.conv2, block1.bn2, block1.relu)
        perm_map = get_layer_perm(subnet1, subnet0, loader)
        permute_output(perm_map, block1.conv2, block1.bn2)
        permute_input(perm_map, block1.conv3)
        permute_output(perm_map, block_sparse.conv2, block_sparse.bn2)
        permute_input(perm_map, block_sparse.conv3)
    
    for k in range(len(blocks0)):
        kk = get_permk(k)
        if kk != last_kk:
            perm_map = get_layer_perm(blocks1[:kk+1], blocks0[:kk+1], loader)
            last_kk = kk
        
        if k > 0:
            permute_output(perm_map, blocks0[k].conv3, blocks0[k].bn3)
            shortcut = blocks0[k].downsample
            if shortcut:
                permute_output(perm_map, shortcut[0], shortcut[1])
            permute_output(perm_map, blocks_sparse[k].conv3, blocks_sparse[k].bn3)
            shortcut = blocks_sparse[k].downsample
            if shortcut:
                permute_output(perm_map, shortcut[0], shortcut[1])
                
        else:
            permute_output(perm_map, model0.conv1, model0.bn1)
            permute_output(perm_map, modelA_sparse.conv1, modelA_sparse.bn1) 
        
        if k+1 < len(blocks0):
            permute_input(perm_map, blocks0[k+1].conv1)
            shortcut = blocks0[k+1].downsample
            if shortcut:
                permute_input(perm_map, shortcut[0])
            permute_input(perm_map, blocks_sparse[k+1].conv1)
            shortcut = blocks_sparse[k+1].downsample
            if shortcut:
                permute_input(perm_map, shortcut[0])
        else:
            permute_input(perm_map, model0.fc)
            permute_input(perm_map, modelA_sparse.fc)
            
    
    return model0, modelA_sparse


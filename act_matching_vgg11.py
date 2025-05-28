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
from models import VGG11_nofc

def run_corr_matrix(net0, net1, loader, device="cpu"):
    """
    Code taken from REPAIR notebook
    Args: net0 -> torch model
          net1 -> torch model which needs to be permuted
    """
    n = len(loader)
    with torch.no_grad():
        net0.eval()
        net1.eval()
        for i, (images, _) in enumerate(tqdm(loader)):
            img_t = images.float().to(device)
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
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)
    assert (row_ind == np.arange(len(corr_mtx_a))).all()
    perm_map = torch.tensor(col_ind).long()
    return perm_map


def get_layer_perm(net0, net1, loader, device):
    corr_mtx = run_corr_matrix(net0, net1, loader, device=device)
    return get_layer_perm1(corr_mtx)


def permute_output(perm_map, layer):
    pre_weights = [layer.weight, layer.bias]
    
    if hasattr(layer, 'weight_orig'):
        pre_weights.append(layer.weight_orig)
    if hasattr(layer, 'weight_mask'):
        pre_weights.append(layer.weight_mask)

    for w in pre_weights:
        w.data = w[perm_map]


# modifies the weight matrix of a layser for a given permutation of the input channels
# works for both conv2d and linear
def permute_input(perm_map, layer):
    w = layer.weight
    w.data = w[:, perm_map]

    if hasattr(layer, 'weight_orig'):
        layer.weight_orig.data = layer.weight_orig[:, perm_map]

    if hasattr(layer, 'weight_mask'):
        layer.weight_mask.data = layer.weight_mask[:, perm_map]

def subnet(model, n_layers):
    return model.features[:n_layers]


def permute_model(model0, model1, modelA_sparse, loader, config):
    """
    function which generates π by matching the activations of model0 to model1 
    args:
    model0 -> torch model
    model1 -> torch model 
    modelA_sparse -> return π(modelA_sparse)
    loader -> dataloader
    returns: π(modelA_sparse) and π(model0)
    """

    device = torch.device(config['device'])
    model0.to(device)
    model1.to(device)
    modelA_sparse.to(device)

    feats1 = modelA_sparse.features
    feats0 = model0.features

    n = len(feats1)
    for i in range(n):
        if not isinstance(feats1[i], nn.Conv2d):
            continue

        # permute the outputs of the current conv layer
        assert isinstance(feats1[i + 1], nn.ReLU)
        perm_map = get_layer_perm(subnet(model1, i + 2), subnet(model0, i + 2), loader, device)

        permute_output(perm_map, feats1[i])
        permute_output(perm_map, feats0[i])

        # look for the next conv layer, whose inputs should be permuted the same way
        next_layer = None
        next_layer0 = None
        for j in range(i + 1, n):
            if isinstance(feats1[j], nn.Conv2d):
                next_layer = feats1[j]
                next_layer0 = feats0[j]
                break
        if next_layer is None:
            next_layer = modelA_sparse.classifier
            next_layer0 = model0.classifier
        permute_input(perm_map, next_layer)
        permute_input(perm_map, next_layer0)

    return modelA_sparse, model0
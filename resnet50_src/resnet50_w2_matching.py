from tqdm import tqdm
import numpy as np
import scipy.optimize

import torch
from torch import nn
import torchvision.transforms as T
import torchvision.models 
from torchvision.models import ResNet50_Weights 
from torch.cuda.amp import autocast
from resnet_torchvision import resnet50_wide

from utils import get_model, yaml, cifar_dataloader, evaluate, check_hooks, calculate_overall_sparsity_from_pth, transfer_sparsity_resnet
import argparse

def check_sparsity_of_mask_A():
    
    # check the sparsity of the mask
    path1 = '/project/def-yani/rjain/narval_cps_march2025/mar2025/model_a_seed11_imp/model_cp/model_level_9_epoch_19.pth'
    device = torch.device("cpu")
    model_A_sparse_test = resnet50_wide()
    model_A_sparse_test = torch.load(path1, map_location = device)
    model_A_sparse_test.to(device)

    print("This is the sparsity of the mask", calculate_overall_sparsity_from_pth(model_A_sparse_test))
    
def get_blocks(net):
        return nn.Sequential(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool),
                            *net.layer1, *net.layer2, *net.layer3, *net.layer4)
    
def run_corr_matrix(net0, net1, loader, device="cuda"):
    """
    Given two networks net0, net1 which each output a feature map of shape NxCxWxH, this will reshape both outputs to (N*W*H)xC 
    and then compute a CxC correlation matrix between the two.
    """

    device = torch.device("cuda")
    net0 = net0.to(device)
    net1 = net1.to(device)

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
                mean0 = torch.zeros_like(mean0_b, device=device)
                mean1 = torch.zeros_like(mean1_b, device=device)
                sqmean0 = torch.zeros_like(sqmean0_b, device=device)
                sqmean1 = torch.zeros_like(sqmean1_b, device=device)
                outer = torch.zeros_like(outer_b, device=device)
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
        # that should be block0.conv2, block0.bn2, it was block1.conv2, block1.bn2 on Jan 9, 2024
        permute_output(perm_map, block0.conv2, block0.bn2)
        # that should be block0.conv3, it was block1.conv3 on Jan 9, 2024
        permute_input(perm_map, block0.conv3)
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


def evaluate_merged_models(model0, permuted_model_1, alpha, config):
    device = config["device"]  # Dynamically determine the device
    model = resnet50_wide().to(device)  # Ensure the new model is on the correct device
    m1, m2 = model0.state_dict(), permuted_model_1.state_dict()

    # Merge state dictionaries and move to the correct device
    sd_alpha = {
        k: (1 - alpha) * m1[k].to(device) + alpha * m2[k].to(device)
        for k in m1.keys()
        if k in m2
    }

    # Load the merged state dictionary into the model
    model.load_state_dict(sd_alpha, strict=False)
    return model

def reset_bn_stats(model, loader, device):
    # Reset stats in BatchNorm layers
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = None  # Use a simple average for running stats
            m.reset_running_stats()

    model.train()  # Set model to training mode

    # Reset stats using the loader
    with torch.no_grad(), autocast():
        for images, _ in loader:
            images = images.to(device)  # Move data to the correct device
            output = model(images.cuda())  # Forward pass to update running stats


def matching():
    with open("/project/def-yani/rjain/sparse-rebasin/sparse-rebasin/configs_imagenet/config_1_80.yaml", "r") as f:
        config = yaml.safe_load(f)
    # this calls the cifar_dataloader but its just bad naming, imagenet is called within it.
    train_dl, test_dl = cifar_dataloader(256, config, args.slurm_tmpdir)

    check_sparsity_of_mask_A()
    
    device1 = torch.device("cuda")

    resnet50_1 = resnet50_wide()
    path1 = "/project/def-yani/rjain/narval_cps_march2025/mar2025/model_a_seed11/model_cp/model_89.pth"
    checkpoint_1 = torch.load(path1, map_location=device1)
    state_dict_1 = {k.replace("module.", ""): v for k, v in checkpoint_1.items()}

    resnet50_2 = resnet50_wide()
    path2 = "/project/def-yani/rjain/narval_cps_march2025/mar2025/model_b_seed33/model_cp/model_89.pth"
    checkpoint_2 = torch.load(path2, map_location=device1)
    state_dict_2 = {k.replace("module.", ""): v for k, v in checkpoint_2.items()}

    resnet50_1.load_state_dict(state_dict_1)
    resnet50_1.to(device1)
    resnet50_2.load_state_dict(state_dict_2)
    resnet50_2.to(device1)
    
    print("Loaded both resnet50 w2 dense models")

    sparse_model_path = '/project/def-yani/rjain/narval_cps_march2025/mar2025/model_a_seed11_imp/model_cp/model_level_9_epoch_19.pth'
    # device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda")
    model_A_sparse = torch.load(sparse_model_path, map_location=device)
    model_A_sparse.to(device)
    
    print("Load the sparse model")

    num_batches = 100

    batches = []
    for i, (images, labels) in enumerate(train_dl):
        if i >= num_batches:
            break
        batches.append((images, labels))

    batch_size = batches[0][0].shape[0]

    total_data_points = num_batches * batch_size

    print(f"Batch size: {batch_size}")
    print(f"Total data points: {total_data_points}")

    permuted_model_1, permuted_model_A_sparse = permute_model_resnet50(resnet50_1, resnet50_2, model_A_sparse, batches, config)
    
    permuted_model_1_path = "/project/def-yani/rjain/sparse-rebasin/sparse-rebasin/temp_resnet50_model_cps/icml_rebuttal_permuted_sparse_and_model_1/permuted_model_1_level_9.pth"
    permuted_mask = "/project/def-yani/rjain/sparse-rebasin/sparse-rebasin/temp_resnet50_model_cps/icml_rebuttal_permuted_sparse_and_model_1/permuted_model_A_sparse_level_9.pth"
    
    torch.save(permuted_model_1, permuted_model_1_path)
    torch.save(permuted_model_A_sparse, permuted_mask)
    
    # Move models to the desired device
    permuted_model_1.to(config["device"])
    permuted_model_A_sparse.to(config["device"])

    # Evaluate both models
    results_1 = evaluate(permuted_model_1, test_dl, config["device"])
    results_2 = evaluate(permuted_model_A_sparse, test_dl, config["device"])

    # Print results
    print(f"Permuted Model 1: Accuracy: {results_1[0]:.2f}%, Loss: {results_1[1]:.4f}")
    print(f"Permuted Model A Sparse: Accuracy: {results_2[0]:.2f}%, Loss: {results_2[1]:.4f}")
    
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    permuted_model_1 = torch.load(permuted_model_1_path, map_location=config["device"])

    model_a = evaluate_merged_models(permuted_model_1, resnet50_2, 0.5, config)

    print('(test_acc, test_loss):')
    print('(α=0.5): %s\t\t<-- Merged model with neuron alignment', evaluate(model_a,test_dl,config["device"]))
    reset_bn_stats(model_a, train_dl, config["device"])
    print('(α=0.5): %s\t\t<-- Merged model with alignment + BN reset', evaluate(model_a,test_dl,config["device"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-slurm_tmpdir", type=str)
    args = parser.parse_args()
    matching()
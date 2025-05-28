import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models import *
from loaders import cifar_dataloader
from torch.nn.utils.prune import custom_from_mask
import math
import copy
import yaml
import os


def evaluate(model, dataloader, device):
    """Evaluate the model on the dataloader, return accuracy and loss. No TTA."""
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            pred = outputs.argmax(dim=1)
            correct += (y == pred).sum().item()
            total += y.size(0)
            loss = loss_fn(outputs, y)
            test_loss += loss.item()
            
    test_accuracy = correct / total
    test_loss /= len(dataloader)
    return test_accuracy * 100, test_loss

def evaluate_top5(model, dataloader, device):
    """Evaluate the model on the dataloader, return top-5 accuracy and loss."""
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    correct_top5 = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            test_loss += loss.item()
            
            # Get the top-5 predictions
            _, pred_top5 = outputs.topk(5, dim=1)
            
            # Check if the correct label is in the top-5 predictions
            correct_top5 += sum([y[i] in pred_top5[i] for i in range(y.size(0))])
            total += y.size(0)

    test_accuracy_top5 = correct_top5 / total 
    test_loss /= len(dataloader)

    return test_accuracy_top5 * 100, test_loss

def evaluate_merged_models(model_A_dense, permuted_model_b, alpha, config):
    device = config["device"]
    model_a = merge_models(model_A_dense, permuted_model_b, alpha, config)   
    return model_a


def check_sparse_gradients(model):
    """
    This is partially incorrect I think; gradients will be zero after optimizer.zero_grad() call. 
    It's currently working before you call zero_call before loss.backwards() in the train_epoch function. 
    It might create an issue in future.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            sparsity = float(torch.sum(param.grad == 0)) / torch.numel(param.grad)
            print(f"Gradient sparsity for {name}: {sparsity:.2f}")

def sparse_grad_tolerance(model, tolerance=1):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_sparsity = float(torch.sum(param.grad == 0)) / torch.numel(param.grad)
            for name, module in model.named_modules():
                if hasattr(param, 'weight_mask'):
                    mask_sparsity, _, _ = calculate_mask_sparsity(module.weight_mask)
                    assert abs(mask_sparsity - grad_sparsity) <= tolerance, \
                        f"Sparsity mismatch in {name}: mask_sparsity={mask_sparsity}, grad_sparsity={grad_sparsity}"

def calculate_mask_sparsity(weight_mask):
    total_params = torch.numel(weight_mask)
    zero_params = torch.sum(weight_mask == 0).item()
    sparsity = zero_params / total_params
    return sparsity, zero_params, total_params

def calculate_overall_sparsity_from_pth(model):
    total_zero_params = 0
    total_params = 0

    for name, param in model.state_dict().items():
        if 'weight_mask' in name:
            _, zero_params, num_params = calculate_mask_sparsity(param)
            total_zero_params += zero_params
            total_params += num_params

    overall_sparsity = total_zero_params / total_params if total_params > 0 else 0
    return overall_sparsity

def calculate_neuron_variances(model, unbiased=False):
    neuron_variances = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() > 1: 
                variances = param.var(dim=list(range(1, param.dim())), unbiased=unbiased)
                neuron_variances[name] = variances
    return neuron_variances

def calculate_neuron_means(model):
    neuron_means = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                means = param.mean(dim=list(range(1, param.dim()))).detach()
                neuron_means[name] = means
    return neuron_means

## population variance
def calculate_layer_variances(model, unbiased=False):
    variances = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                var = param.var(unbiased=unbiased).item() 
                variances[name] = var
    return variances

def calculate_layer_means(model):
    means = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                mean = param.mean().item()
                means[name] = mean
    return means

def get_fan_in_tensor_mike(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() < 2:
        raise ValueError(
            "Fan in can not be computed for tensor with fewer than 2 dimensions"
        )
    if mask.dtype == torch.bool:
        fan_in_tensor = mask.sum(dim=list(range(1, mask.dim())))
    else:
        fan_in_tensor = (mask != 0.0).sum(dim=list(range(1, mask.dim())))
    return fan_in_tensor

def scale_model_with_fan_in(B_init_dense, A_sparse, config, mode: str = "fan_in", neuron_basis = True):
    if mode.lower() != "fan_in":
        raise NotImplementedError(
            "Only mode==`fan_in` has currently been implemented at this time."
        )
    
    if neuron_basis:
        layer_variances = calculate_neuron_variances(B_init_dense)
        layer_means = calculate_neuron_means(B_init_dense)

    else:
        layer_variances = calculate_layer_variances(B_init_dense)
        layer_means = calculate_layer_means(B_init_dense) 

    print("print layer variance keys:", layer_variances.keys())
    print("print layer means keys:", layer_means.keys())

    for (name_B, m_B), (name_A, m_A) in zip(B_init_dense.named_modules(), A_sparse.named_modules()):
        if isinstance(m_B, (nn.Conv2d, nn.Linear)) and isinstance(m_A, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                if config['model_type'] == 'ResNet':
                    if not hasattr(m_A, 'weight_mask'):
                        print(f"Skipping layer {name_A} ({m_A}) as it does not have weight_mask.")
                        continue
                mask = m_A.weight_mask
                if mask.shape != m_B.weight.shape:
                    raise ValueError(f"Sparsity mask and weight tensor shape do not match for layer {name_A}!")
                if 0 in m_B.weight.shape:
                    print(f"Initializing zero-element tensors is a no-op for layer {name_B}")
                    return m_B.weight
                fan_in = get_fan_in_tensor_mike(mask)
                
                weight_name = f"{name_B}.weight_orig"
                
                if weight_name not in layer_means or weight_name not in layer_variances:
                    print(f"Layer {name_B} not found in layer_means or layer_variances.")
                    continue
                
                mean_B = layer_means[weight_name]
                variance_B = layer_variances[weight_name]

                # Calculate the target variance
                target_variance = 2 / fan_in
                if neuron_basis:
                    for i in range(m_B.weight.shape[0]): 
                        if fan_in[i] != 0:
                            
                            m_B.weight.data[i] -= mean_B[i]
                            sqrt_target_variance = math.sqrt(target_variance[i]) / math.sqrt(variance_B[i])
                            m_B.weight.data[i] *= sqrt_target_variance
                            
                        elif fan_in[i] == 0:
                            m_B.weight.data[i] = 0
                else:
                    for i in range(m_B.weight.shape[0]):
                        if fan_in[i] != 0:

                            m_B.weight.data[i] -= mean_B
                            sqrt_target_variance = math.sqrt(target_variance[i]) / math.sqrt(variance_B)
                            m_B.weight.data[i] *= sqrt_target_variance

                        elif fan_in == 0:
                            m_B.weight.data[i] = 0
    
    return B_init_dense

def compute_loss_and_backward(model, data, target, loss_fn):
    model.train()
    output = model(data)
    loss = loss_fn(output, target)
    model.zero_grad()
    loss.backward()
    return loss

def calculate_gradient_norms(model):
    grad_norms = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module.weight.grad is not None:
                grad_norms[f"{name}.weight"] = module.weight.grad.norm().item()
    return grad_norms

def compute_gradient_norms(scaled_model, train_dl, config):
    
    for data, target in train_dl:
        data, target = data.to(config["device"]), target.to(config["device"])
        break 
    
    loss_fn = nn.CrossEntropyLoss()
    compute_loss_and_backward(scaled_model, data, target, loss_fn)
    grad_norms = calculate_gradient_norms(scaled_model)

    print("Gradient norms after scaling and resetting BN stats:", grad_norms)

def transfer_sparsity_adnan(model_A, model_B):
    ''' this code will only work  for the vgg model def'''

    # transfer features mask first
    modules_to_replace = {}

    for (name_A, module_A), (name_B, module_B) in zip(model_A.features.named_modules(), model_B.features.named_modules()):
        assert(type(module_A) is type(module_B) and name_A == name_B) # and hasattr(module_A, 'weight_mask'))

        if name_A != 'classifier' and hasattr(module_A, "weight_mask"):
            print('Replacing layer in model B with masked layer:', name_A)
            modules_to_replace[name_A] = custom_from_mask(copy.deepcopy(module_B), name="weight", mask=module_A.weight_mask)

        else:
            print('Skipping layer in model A when copying masks:', name_A)

    print(modules_to_replace)
    for module_name, module in modules_to_replace.items():
        with torch.no_grad():
            assert(hasattr(model_B.features, module_name))
            setattr(model_B.features, module_name, module)
            assert(hasattr(getattr(model_B.features, module_name), 'weight_mask'))

    # transfer classifier mask
    modules_to_replace = {}
    for (name_A, module_A), (name_B, module_B) in zip(model_A.classifier.named_modules(), model_B.classifier.named_modules()):
        assert(type(module_A) is type(module_B) and name_A == name_B) # and hasattr(module_A, 'weight_mask'))

        if hasattr(module_A, "weight"):
            print('Replacing layer in model B with masked layer:', name_A)
            modules_to_replace[name_A] = custom_from_mask(copy.deepcopy(module_B), name="weight", mask=module_A.weight_mask)
        else:
            print('Skipping layer in model A when copying masks:', name_A)

    print(modules_to_replace)
    for module_name, module in modules_to_replace.items():
        with torch.no_grad():
            setattr(model_B, 'classifier', module)
            assert(hasattr(getattr(model_B, 'classifier'), 'weight_mask'))

def check_hooks(model): 
    def check_module_hooks(module): 
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)): 
            print(module, module._forward_pre_hooks) 
            if len(module._forward_pre_hooks) == 0: 
                print('Error') 
                
        for child in module.children(): 
            check_module_hooks(child) 
    
    check_module_hooks(model)
    
def get_model(config):
    if config["model_type"] == "VGG11":
        return VGG11_nofc("VGG11", config, init_weights=config["init_weights"]).to(config["device"])
    elif config["model_type"] == "ResNet":
        return ResNet(BasicBlock, [3, 3, 3], config, w=config["width_multiplier"]).to(config["device"])
    elif config["model_type"] == "ResNet50":
        return resnet50().to(config["device"])
    else:
        raise ValueError(f"Unsupported model type: {config['model_type']}")
    
def merge_models(m0, m1, alpha, config):
    net = get_model(config)
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    sd_alpha = {
        k: (1 - alpha) * sd0[k].to(config['device']) + alpha * sd1[k].to(config['device'])
        for k in sd0.keys()
        if k in sd1
    }
    net.load_state_dict(sd_alpha, strict=False)
    return net

"""
The following function is taken from REPAIR.
"""
# use the train loader with data augmentation as this gives better results
def reset_bn_stats(model, loader):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None # use simple average
            m.reset_running_stats()
    # run a single train epoch with augmentations to recalc stats
    model.train()
    with torch.no_grad(), autocast():
        for images, _ in loader:
            output = model(images.cuda())

def reset_bn_stats_sparse(model, loader, iter = 40, reset_params=False):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = 0.1 # use simple average
            m.reset_running_stats()
            if reset_params == True:
                m.reset_parameters()
                
    # run a single train epoch with augmentations to recalc stats
    model.train()
    with torch.no_grad():
        for epoch in range(iter):
            for images, _ in loader:
                output = model(images.cuda())

def compare_model_weights(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if "bias" not in name1 and "bias" not in name2:
            if not torch.equal(param1, param2):
                print(f"Difference found in layer: {name1}")
                return False
    print("All weights match for all layers.")
    return True

def transfer_sparsity_resnet(model_A, model_B):
    '''
    function to transfer sparsity for resnet model definition as per the REPAIR codebase
    args:
        model_A: sparse model with torch pruner mask/weight_orig
        model_B: dense model on which mask needs to applied
    return:
        modified model_B with mask applied in-place
    '''

    
    modules_to_replace = {}
    
    
    for (name_A, module_A), (name_B, module_B) in zip(model_A.named_modules(), model_B.named_modules()):
        if type(module_A) is not type(module_B) or name_A != name_B:
            print(f"Mismatch found:")
            print(f"- model_A: {name_A}, type: {type(module_A)}")
            print(f"- model_B: {name_B}, type: {type(module_B)}")
        assert(type(module_A) is type(module_B) and name_A == name_B)
        
        if hasattr(module_A, "weight_mask"):
            print('Replacing layer in model B with masked layer:', name_A)
            modules_to_replace[name_A] = custom_from_mask(copy.deepcopy(module_B), name="weight", mask=module_A.weight_mask)
        else:
            print('Skipping layer in model A when copying masks:', name_A)
    
    for module_name, module in modules_to_replace.items():
        with torch.no_grad():
            mod_attr = model_B
            for i in range(len(module_name.split('.'))-1):
                if len(module_name.split('.')[i])!=1:
                    mod_attr = getattr(model_B, module_name.split('.')[i])
                else:
                    mod_attr = mod_attr[int(module_name.split('.')[i])]
                
            assert(hasattr(mod_attr, module_name.split('.')[-1]))
            setattr(mod_attr, module_name.split('.')[-1], module)
            assert(hasattr(getattr(mod_attr, module_name.split('.')[-1]), 'weight_mask'))

class TrackLayer(nn.Module):
    def __init__(self, layer, one_d=False):
        super().__init__()
        self.layer = layer
        dim = layer.conv3.out_channels
        self.bn = nn.BatchNorm2d(dim)
        
    def get_stats(self):
        return (self.bn.running_mean, self.bn.running_var.sqrt())
        
    def forward(self, x):
        x1 = self.layer(x)
        self.bn(x1)
        return x1

class ResetLayer(nn.Module):
    def __init__(self, layer, one_d=False):
        super().__init__()
        self.layer = layer
        dim = layer.conv3.out_channels
        self.bn = nn.BatchNorm2d(dim)
        
    def set_stats(self, goal_mean, goal_std):
        self.bn.bias.data = goal_mean
        self.bn.weight.data = goal_std
        
    def forward(self, x):
        x1 = self.layer(x)
        return self.bn(x1)

# adds TrackLayer around each block
def make_tracked_net(net, config):
    net1 = get_model(config)
    net1.load_state_dict(net.state_dict())
    for i in range(4):
        layer = getattr(net1, 'layer%d' % (i+1))
        for j, block in enumerate(layer):
            layer[j] = TrackLayer(block).cuda()
    return net1

# adds ResetLayer around each block
def make_repaired_net(net, config):
    net1 = get_model(config)
    net1.load_state_dict(net.state_dict())
    for i in range(4):
        layer = getattr(net1, 'layer%d' % (i+1))
        for j, block in enumerate(layer):
            layer[j] = ResetLayer(block).cuda()
    return net1
            
def repair_merged_model(permuted_model_0, model_1, model_a, train_dl, config, alpha=0.5):
    # Evaluate merged models
    model0 = evaluate_merged_models(permuted_model_0, model_1, 0, config)
    model1 = evaluate_merged_models(permuted_model_0, model_1, 1, config)

    # Calculate all neuronal statistics in the endpoint networks
    wrap0 = make_tracked_net(model0)
    wrap1 = make_tracked_net(model1)
    reset_bn_stats(wrap0, train_dl)
    reset_bn_stats(wrap1, train_dl)

    wrap_a = make_repaired_net(model_a)
    # Iterate through corresponding triples of (TrackLayer, TrackLayer, ResetLayer)
    # around conv layers in (model0, model1, model_a).
    for track0, track1, reset_a in zip(wrap0.modules(), wrap1.modules(), wrap_a.modules()): 
        if not isinstance(track0, TrackLayer):
            continue  
        assert (isinstance(track0, TrackLayer)
                and isinstance(track1, TrackLayer)
                and isinstance(reset_a, ResetLayer))

        # Get neuronal statistics of original networks
        mu0, std0 = track0.get_stats()
        mu1, std1 = track1.get_stats()
        # Set the goal neuronal statistics for the merged network 
        goal_mean = (1 - alpha) * mu0 + alpha * mu1
        goal_std = (1 - alpha) * std0 + alpha * std1
        reset_a.set_stats(goal_mean, goal_std)

    # Estimate mean/vars such that when added BNs are set to eval mode,
    # neuronal stats will be goal_mean and goal_std.
    reset_bn_stats(wrap_a, train_dl)
    
    return wrap_a

def update_config_with_slurm_tmpdir(config, slurm_tmpdir):
    config['dataset_root']['imagenet'] = os.path.join(slurm_tmpdir, 'imagenet')
    return config
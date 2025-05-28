import torch
import os
import torch.nn.utils.prune as prune
import wandb
import torch.optim as optim
from training import train_epoch
from loaders import cifar_dataloader
from utils import evaluate, calculate_overall_sparsity_from_pth, check_sparse_gradients, calculate_mask_sparsity, get_model
import yaml

def prune_model(
    config,
    model,
    target_sparsity,
    optimizer_config,
    prune_epochs,
    initial_lr,
    batch_size,
    device,
    initial_prune_perc=0.2,
    train_epochs_per_prune=5,
    sparsity_tolerance=0.005, 
):
    total_epochs = 0
    train_dl, test_dl = cifar_dataloader(batch_size, config)
    
    if config["model_type"] == "VGG11":
        parameters_to_prune = [
            (module, 'weight') for name, module in model.named_modules() 
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
        ]
    elif config["model_type"] == "ResNet":
        parameters_to_prune = [
            (module, 'weight') for name, module in model.named_modules() 
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and 'shortcut' not in name
        ]
    elif config["model_type"] == "ResNet50":
        parameters_to_prune = [
            (module, 'weight') for name, module in model.named_modules() 
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and 'shortcut' not in name and 'downsample' not in name
        ]
    else:
        raise ValueError(f"Unsupported model type: {config['model_type']}")

    current_sparsity = calculate_overall_sparsity_from_pth(model)
    print(f"Initial sparsity: {current_sparsity*100:.2f}%")

    for epoch in range(prune_epochs):
        optimizer_type = getattr(optim, optimizer_config["type"])
        optimizer = optimizer_type(
            model.parameters(), lr=initial_lr, **optimizer_config.get('params', {})
        )
        lrs = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, train_epochs_per_prune, eta_min=0.0
        )

        current_sparsity = calculate_overall_sparsity_from_pth(model)
        print(f"Current sparsity before pruning: {current_sparsity*100:.2f}%")
        
        if current_sparsity < target_sparsity:
            max_additional_sparsity = target_sparsity - current_sparsity
            adjusted_prune_perc = min(
                initial_prune_perc, max_additional_sparsity / (1 - current_sparsity)
            )
        else:
            adjusted_prune_perc = 0

        print(f"Adjusted prune percentage: {adjusted_prune_perc*100:.2f}%")

        prune.global_unstructured(
            parameters_to_prune, 
            pruning_method=prune.L1Unstructured, 
            amount=adjusted_prune_perc
        )

        for epoch_num in range(train_epochs_per_prune):
            train_loss_est = train_epoch(model, train_dl, optimizer, device)
            test_accu, test_loss = evaluate(model, test_dl, device)
            print(
                f"Prune Epoch: {epoch:2d}, Train Epoch: {epoch_num} - "
                f"Test accuracy: {test_accu:.2f} - Test loss: {test_loss:.4f} - "
                f"Train loss est.: {train_loss_est:.4f} - Learning rate: {optimizer.param_groups[0]['lr']:.4f}"
            )
            wandb.log(
                {
                    "Model A Sparse Prune Epoch": epoch + 1,
                    "Model A Sparse Train Loss": train_loss_est,
                    "Model A Sparse Test Loss": test_loss,
                    "Model A Sparse Test Accuracy": test_accu,
                }
            )
            total_epochs += 1
            lrs.step()

        overall_sparsity = calculate_overall_sparsity_from_pth(model)
        print(f"Prune Epoch: {epoch} - Overall Sparsity: {overall_sparsity*100:.2f}%")

        for name, param in model.state_dict().items():
            if 'weight_mask' in name:
                sparsity, zero_params, total_params = calculate_mask_sparsity(param)
                print(f"Layer {name} - Sparsity: {sparsity*100:.2f}% - Zero params: {zero_params} - Total params: {total_params}")

        check_sparse_gradients(model)
        
        if overall_sparsity >= target_sparsity - sparsity_tolerance:
            print(f"Target sparsity of {target_sparsity*100:.2f}% reached. Terminating pruning.")
            break
    
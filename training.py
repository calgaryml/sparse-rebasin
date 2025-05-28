import torch
import torch.nn as nn
import torch.optim as optim
import math
from loaders import cifar_dataloader
from utils import evaluate, sparse_grad_tolerance
import wandb
import yaml


def train_epoch(model, dataloader, optimizer, device, sparse=False):

    model.train()
    running_loss = 0
    loss_fn = nn.CrossEntropyLoss()

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        running_loss += loss.item()
        # Backpropagation
        loss.backward()
        
        if sparse:
            sparse_grad_tolerance(model)
        optimizer.step()

    return running_loss / len(dataloader)


def trainer(model, optimizer_config, epochs, batch_size, device, model_name, config, sparse=False, training_type="naive", k=0):

        
    optimizer_type = getattr(optim, optimizer_config["type"]) 
    lr = optimizer_config["sparse_lr"] if sparse else optimizer_config["lr"]
    optimizer = optimizer_type(
        model.parameters(),
        lr=lr,
        momentum=optimizer_config["momentum"],
        weight_decay=optimizer_config["weight_decay"],
    )
    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0)
    train_dl, test_dl = cifar_dataloader(batch_size, config)

    for epoch in range(epochs):
        train_loss_est = train_epoch(model, train_dl, optimizer, device, sparse=sparse)
        if epoch%5==0 and 'Permuted' not in model_name and 'Naive' not in model_name and 'LTH' not in model_name: 
            torch.save(model.state_dict(), config['save_path'] + model_name + '_sparsity_' + str(config['pruning']['sparsity'])+ '_seed_' + str(config['seed']) + '_epoch_' + str(epoch))
        test_accu, test_loss = evaluate(model, test_dl, device)
        print(
            f"Epoch: {epoch+1:2d} - Test accuracy: {test_accu:.2f} - Test loss: {test_loss:.4f} - ",
            f"Train loss est.: {train_loss_est:.4f} - Learning rate: {optimizer.param_groups[0]['lr']:.4f}",
        )
        if sparse:
            if training_type == "naive":
                wandb.log(
                    {
                        f"{model_name} Naive Epoch": epoch + 1,
                        f"{model_name} Naive Train Loss": train_loss_est,
                        f"{model_name} Naive Test Loss": test_loss,
                        f"{model_name} Naive Test Accuracy": test_accu,
                    }
                )
            elif training_type == "permuted":
                wandb.log(
                    {
                        f"{model_name}  Epoch": epoch + 1,
                        f"{model_name}  Train Loss": train_loss_est,
                        f"{model_name}  Test Loss": test_loss,
                        f"{model_name}  Test Accuracy": test_accu,
                    }
                )
            elif training_type == "LTH":
                wandb.log(
                    {
                        f"{model_name}  Epoch": epoch + 1,
                        f"{model_name}  Train Loss": train_loss_est,
                        f"{model_name}  Test Loss": test_loss,
                        f"{model_name}  Test Accuracy": test_accu,
                    }
                )
        else:
            wandb.log(
                {
                    f"{model_name}  Epoch": epoch + 1,
                    f"{model_name}  Train Loss": train_loss_est,
                    f"{model_name}  Test Loss": test_loss,
                    f"{model_name}  Test Accuracy": test_accu,
                }
            )
        lrs.step()
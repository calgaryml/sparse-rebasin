import torch
from torchvision import datasets, transforms
import os
import yaml

def cifar_dataloader(batch_size, config, slurm_tmpdir):

    dataset = config["dataset"]
    dataset_root = config["dataset_root"][dataset]

    if dataset == "cifar10":
        print("Load CIFAR10 dataset")
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_ds = datasets.CIFAR10(
            root=dataset_root,
            train=True,
            download=False,
            transform=train_transform,
        )
        test_ds = datasets.CIFAR10(
            root=dataset_root,
            train=False,
            download=False,
            transform=test_transform,
        )
    elif dataset == "cifar100":
        print("Load CIFAR100 dataset")
        normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_ds = datasets.CIFAR100(
            root=dataset_root,
            train=True,
            download=False,
            transform=train_transform,
        )
        test_ds = datasets.CIFAR100(
            root=dataset_root,
            train=False,
            download=False,
            transform=test_transform,
        )
        
    elif dataset == "imagenet":
        print("Load ImageNet dataset")
        traindir = os.path.join(slurm_tmpdir, 'train')
        valdir = os.path.join(slurm_tmpdir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_ds = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        test_ds = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        
    elif dataset == "svhn":
        print("Load SVHN dataset")
        normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_ds = datasets.SVHN(
            root=dataset_root,
            split="train",
            download=False,
            transform=train_transform,
        )
        test_ds = datasets.SVHN(
            root=dataset_root,
            split="test",
            download=False,
            transform=test_transform,
        )      
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, test_loader
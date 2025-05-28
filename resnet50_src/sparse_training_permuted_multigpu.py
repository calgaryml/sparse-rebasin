import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import wandb
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import datetime

#import torchvision.datasets as datasets
# import torchvision.models as models
# import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from models import resnet50

import torch.nn.utils.prune as prune
import torch.optim as optim
from utils import evaluate, calculate_overall_sparsity_from_pth, check_sparse_gradients, calculate_mask_sparsity, get_model, transfer_sparsity_resnet, check_hooks
import copy
from resnet_torchvision import resnet50_wide


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                     help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--prune', action='store_true', help="prune the model")
# parser.add_argument('--prune_level',type=int)
parser.add_argument('--rewind_epoch',type=int) ## 0, 10, 25, 50
parser.add_argument('--sparsity',type=int) #10 is 90% sparsity, 13 is 95% sparsity, 16 is 97% sparsity


best_acc1 = 0


### PERMUTED (FILE PATHS ARE MINNEWANKA)

def main():
    args = parser.parse_args()


    if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
            warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    dense_model_trained_for_these_many_epochs = 89 
    k = args.rewind_epoch    
    epochs = dense_model_trained_for_these_many_epochs - k
    
    print("\n We are sparse training the permuted for these many epochs: ", epochs)

    device_for_loading_perm = torch.device('cpu')

    model_B_dense_init_perm = resnet50_wide().to(device_for_loading_perm)
    path_rewound_point_B_for_naive_and_permuted = f"/scratch/rjain/sparse_training/wide_resnet50_cp/model_b/model_{k}.pth"

    checkpoint_1 = torch.load(path_rewound_point_B_for_naive_and_permuted, map_location=device_for_loading_perm)
    state_dict_1 = {k.replace("module.", ""): v for k, v in checkpoint_1.items()}


    model_B_dense_init_perm.load_state_dict(state_dict_1)

    perm_model_A_sparse_path = f"/scratch/rjain/sparse_training/wide_resnet50_cp/permuted_mask/permuted_model_A_sparse_jan20_1000_batches.pth"
    permuted_model_A_sparse = torch.load(perm_model_A_sparse_path, map_location=device_for_loading_perm)

    transfer_sparsity_resnet(permuted_model_A_sparse, model_B_dense_init_perm)
    check_hooks(model_B_dense_init_perm)
    print("Sparsity of the rewound init after transfer sparsity: ",calculate_overall_sparsity_from_pth(model_B_dense_init_perm))

    if args.sparsity == 7:
        sparsity_for_printing = 80
    elif args.sparsity == 10:
        sparsity_for_printing = 90
    elif args.sparsity == 13:
        sparsity_for_printing = 95
    else:
        sparsity_for_printing = 97

    torch.save(model_B_dense_init_perm, f"perm_feb19_rewind_at_{args.rewind_epoch}_sparsity_{sparsity_for_printing}")


    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, epochs))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, epochs)







def main_worker(gpu, ngpus_per_node, args, sparse_epochs):
    global best_acc1
    args.gpu = gpu


    #### IMPORTANT TO CHANGE THE LR FOR SPARSE TRAINING.

    def set_lr(ep):
        '''
        function to set the learning rate
        based on the ResNet paper. 
        Linear warm up for first 5 epochs, followed by step lr

        ep: epoch number

        returns the lr

        '''
        if ep <= 5:
            return ((args.lr - 1e-2)/5)*(ep)

        
        if ep>5:
            ratio = (ep//31)
            return args.lr * (1/10**ratio)
    
    # if args.sparsity == 7:
    #     sparsity_for_printing = 80
    # elif args.sparsity == 10:
    #     sparsity_for_printing = 90
    # elif args.sparsity == 13:
    #     sparsity_for_printing = 95
    # else:
    #     sparsity_for_printing = 97

    if gpu == 0:
        wandb.login()
        current_time = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

        if args.sparsity == 7:
            sparsity_for_printing = 80
        elif args.sparsity == 10:
            sparsity_for_printing = 90
        elif args.sparsity == 13:
            sparsity_for_printing = 95
        else:
            sparsity_for_printing = 97    

        wandb.init(
        project="rohan_resnet50_imagenet_perm",
            name=f"resnet50_perm_{args.rewind_epoch}_sparsity_{sparsity_for_printing}; {current_time}",
            config={
                "rewind_epoch": args.rewind_epoch,
                "lr": args.lr,
                "sparsity": args.sparsity,
    
            }
        )
        print('wandb initialized')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.sparsity == 7:
        sparsity = 80
    elif args.sparsity == 10:
        sparsity = 90
    elif args.sparsity == 13:
        sparsity = 95
    else:
        sparsity = 97

    model = torch.load(f"perm_feb19_rewind_at_{args.rewind_epoch}_sparsity_{sparsity}") # model is being passed as an arg now
    #model.load(prunecheckpoint)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")



    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         if args.gpu is None:
    #             checkpoint = torch.load(args.resume)
    #         elif torch.cuda.is_available():
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.gpu)
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.gpu)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         # scheduler.load_state_dict(checkpoint['scheduler'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
            traindir,
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))

    val_dataset = torchvision.datasets.ImageFolder(
            valdir,
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None 

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)


    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

          
    print("\n Starting sparse training for {} epochs".format(sparse_epochs))
    for epoch in range(args.start_epoch, sparse_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)


        for g in optimizer.param_groups:
            g['lr'] = set_lr(epoch+1)/10

        train(train_loader, model, criterion, optimizer, epoch, device, args)

        acc1, acc5 = validate(val_loader, model, criterion, args)
        top1_acc = acc1
        top5_acc = acc5

        if gpu == 0:  
            wandb.log({
                "epoch": epoch + 1,
                "top1_acc": top1_acc,
                "top5_acc": top5_acc
            })
            check_sparse_gradients(model)

        # if gpu==0:
        #     # wandb.log({"test_acc": acc1, "lr_fine":set_lr(epoch+1)})

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

    if gpu == 0:
        wandb.log({
            "final_top1_acc": top1_acc,
            "final_top5_acc": top5_acc
        })
        print(f"Training complete. Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%")
        print(f"This is the accuracy for the rewind point {args.rewind_epoch} at sparsity {sparsity_for_printing}%.")
        # torch.save(model, "/project/def-yani/rjain/sparse-rebasin/sparse-rebasin/sparse_solutions_w2_sparsity_90/Permuted_solution_w2_sparsity_90.pth")

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #         # 'scheduler' : scheduler.state_dict()
        #     }, is_best)

        # if gpu==0:
        #     check_sparse_gradients(model)
        #     model_a = copy.deepcopy(model)
        #     new_model = model_a.module
        #     new_model = new_model.cpu()
        #     torch.save(new_model, '/home/rjain/scratch/resnet50/imp/nov15/model_cp/'+ 'model_level_{}_epoch_{}.pth'.format(prune_level,epoch))
        #     torch.save(optimizer.state_dict(),'/home/rjain/scratch/resnet50/imp/nov15/optim_cp/'+ 'optim_level_{}_epoch_{}.pth'.format(prune_level,epoch))



def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    
def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg, top5.avg


# def save_checkpoint(state, is_best, filename='/home/rjain/scratch/resnet50/sparse_training/baseline/checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, '/home/rjain/scratch/resnet50/sparse_training/baseline/model_best.pth.tar')



class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

import copy
from training import *
from models import *
from utils import *
import yaml
from act_matching_vgg11 import permute_model
from act_matching_resnet20 import permute_model_resnet
from resnet50_src.act_matching_resnet50 import permute_model_resnet50
import datetime
from pruning import prune_model
import argparse
import numpy as np
import random
import os

def train_model_a_b(seed, config):
    '''
    function to train model_a_dense, model_a_sparse (imp) and model_b dense,
    and saves all checkpoints.
    '''

    results = {}

    current_time = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

    # Check selected backend is supported
    assert hasattr(torch.backends, config["device"]), f"Device {config['device']} not supported"
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Initialize wandb with a project name and a unique run name
    wandb.init(
        project=config["project"],
        entity=config["entity"],
        name=f"{config['model_type']}; seed_{config['seed']}; {current_time}",
        config={'sparsity':config['pruning']['sparsity'],
                'width':config['width_multiplier'],
                }
    )


    ############################################################## Dense training: Model A ########################################################
    model_A_dense = get_model(config)
    trainer(model_A_dense, config["optimizer"], config["epochs"], config["batch_size"], config["device"], "Model_A_Dense", sparse=False, config=config)
    train_dl, test_dl = cifar_dataloader(config["batch_size"], config)

    results["A_dense"] = evaluate(
        model_A_dense, test_dl, config["device"]
    )
    print(f"A_dense: {results['A_dense'][0]:.2f}%, {results['A_dense'][1]:.4f}")
    
    if config['model_type'] == 'ResNet50':
        results["A_dense_top5"] = evaluate_top5(model_A_dense, test_dl, config["device"])
        print(f"A_dense_top5: {results['A_dense_top5'][0]:.2f}%, Loss: {results['A_dense_top5'][1]:.4f}")
    ###############################################################################################################################################


    ############################################################## Pruning Model A after training #################################################
    if config['model_type'] == 'ResNet' or config['model_type'] == 'VGG11' or config['model_type'] == 'ResNet50':
        model_A_sparse = copy.deepcopy(model_A_dense)
        if config['model_type'] == 'ResNet' or config['model_type'] == 'ResNet50':
            sparsity_tolerance = config['pruning']['tolerance_resnet20']
        else:
            sparsity_tolerance = config['pruning']['tolerance_vgg11']
        prune_model(
            config,
            model=model_A_sparse,
            target_sparsity=config['pruning']['sparsity'],
            optimizer_config=config['optimizer'],
            prune_epochs=config['pruning']['prune_epochs'],
            initial_lr=config['pruning']['prune_lr'],
            batch_size=config['batch_size'],
            device=device,  
            initial_prune_perc=config['pruning']['prune_perc'],
            train_epochs_per_prune=config['pruning']['train_epochs_per_prune'],
            sparsity_tolerance=sparsity_tolerance,
        )
        original_model_A_sparse = copy.deepcopy(model_A_sparse)
        torch.save(model_A_sparse, config['save_path'] + 'model_a_sparse_sparsity_' + str(config['pruning']['sparsity'])+ '_seed_' + str(config['seed']) )
        print("\n First hooks check \n")
        check_hooks(original_model_A_sparse)


        for name, module in model_A_sparse.named_modules():
            if isinstance(module, nn.Conv2d):
                if config['model_type'] == 'ResNet' or config['model_type'] == 'ResNet50' and 'shortcut' in name and 'downsample' in name:
                    continue
                assert hasattr(module, 'weight_mask')
        results["A_sparse"] = evaluate(
            model_A_sparse, test_dl, config["device"]
        )
        print(f"A_sparse: {results['A_sparse'][0]:.2f}%, {results['A_sparse'][1]:.4f}")
    ###############################################################################################################################################
    
    ############################################################## Dense training: Model B ########################################################
    model_B_dense = get_model(config)
    trainer(model_B_dense, config["optimizer"], config["model_b_epochs"], config["batch_size"], config["device"], "Model_B_Dense", sparse=False, config=config)
    results["B_dense"] = evaluate(
        model_B_dense, test_dl, config["device"]
    )
    print(f"B_dense: {results['B_dense'][0]:.2f}%, {results['B_dense'][1]:.4f}")
    
    if config['model_type'] == 'ResNet50':
        results["B_dense_top5"] = evaluate_top5(model_A_dense, test_dl, config["device"])
        print(f"B_dense_top5: {results['B_dense_top5'][0]:.2f}%, Loss: {results['B_dense_top5'][1]:.4f}")
    ###############################################################################################################################################
    
    if config['model_type'] == 'ResNet':
        permute_model_0, permuted_model_A_sparse = permute_model_resnet(model_A_dense, model_B_dense, model_A_sparse, train_dl, config)
        results["A_dense_permuted"] = evaluate(permute_model_0, test_dl, config["device"])
        print(f"A_dense_permuted: {results['A_dense_permuted'][0]:.2f}%, {results['A_dense_permuted'][1]:.4f}")
        
        print("\n Pre-reset")
        model_a = evaluate_merged_models(permute_model_0, model_B_dense, 0.5, config)
        train_accuracy, train_loss = evaluate(model_a, train_dl, config["device"])
        test_accuracy, test_loss = evaluate(model_a, test_dl, config["device"])
        print(
            f"(α=0.5): <-- Merged Model\tTrain Accuracy: {train_accuracy}\tTrain Loss: {train_loss:.4f}\t"
            f"Test Accuracy: {test_accuracy}\tTest Loss: {test_loss:.4f}"
        )
        
        print("\n Post-reset")
        reset_bn_stats(model_a, train_dl)
        results["Merged Model Post reset evaluation"] = evaluate(model_a, test_dl, config["device"])
        print(f"Merged Model Post reset evaluation: {results['Merged Model Post reset evaluation'][0]:.2f}%, {results['Merged Model Post reset evaluation'][1]:.4f}")

        results["A_sparse_permuted"] = evaluate(
            permuted_model_A_sparse, test_dl, config["device"]
        )
        print(
            f"\n A_sparse_permuted: {results['A_sparse_permuted'][0]:.2f}%, {results['A_sparse_permuted'][1]:.4f}"
        )

        if compare_model_weights(original_model_A_sparse, permuted_model_A_sparse):
            print("\n\n The weights match for all layers.")
        else:
            print("\n\n The weights do not match for all layers.")
    
    elif config['model_type'] == 'VGG11':
        
        #### π(modelA_sparse) ###
        permuted_model_A_sparse, permuted_model_a_dense = permute_model(
            model_A_dense, model_B_dense, model_A_sparse, train_dl, config
        )
        for name, module in permuted_model_A_sparse.named_modules():
            if isinstance(module, nn.Conv2d):
                assert hasattr(module, 'weight_mask')
        check_hooks(permuted_model_A_sparse)
        results["A_sparse_permuted"] = evaluate(
            permuted_model_A_sparse, test_dl, config["device"]
        )
        print(
            f"A_sparse_permuted: {results['A_sparse_permuted'][0]:.2f}%, {results['A_sparse_permuted'][1]:.4f}"
        )
        
    
        #### Check strength of permutation by interpolating between dense model B and dense permuted model A ####
        results["A_dense_permuted"] = evaluate(
            permuted_model_a_dense, test_dl, config["device"]
        )
        print(
            f"A_dense_permuted: {results['A_dense_permuted'][0]:.2f}%, {results['A_dense_permuted'][1]:.4f}"
        )
        model_a = evaluate_merged_models(model_B_dense, permuted_model_a_dense, 0.5, config)
        train_accuracy, train_loss = evaluate(model_a, train_dl, config["device"])
        test_accuracy, test_loss = evaluate(model_a, test_dl, config["device"])
        print(
            f"(α=0.5): <-- Merged Model\tTrain Accuracy: {train_accuracy}\tTrain Loss: {train_loss:.4f}\t"
            f"Test Accuracy: {test_accuracy}\tTest Loss: {test_loss:.4f}"
        )

        if compare_model_weights(original_model_A_sparse, permuted_model_A_sparse):
            print("\n\n The weights match for all layers.")
        else:
            print("\n\n The weights do not match for all layers.")
    
    elif config['model_type'] == 'ResNet50':
        num_batches = 50
        batches = []
        for i, (images, labels) in enumerate(train_dl):
            if i >= num_batches:
                break
            batches.append((images, labels))
        batch_size = batches[0][0].shape[0]
        total_data_points = num_batches * batch_size
        print(f"Batch size: {batch_size}")
        print(f"Total data points: {total_data_points}")

        permute_model_0, permuted_model_A_sparse = permute_model_resnet50(model_A_dense, model_B_dense, model_A_sparse, batches, config)
        torch.save(permute_model_0, config['save_path'] + 'permuted_model_0_sparsity_' + str(config['pruning']['sparsity']) + '_seed_' + str(config['seed']))
        results["A_dense_permuted"] = evaluate(permute_model_0, test_dl, config["device"])
        print(f"A_dense_permuted: {results['A_dense_permuted'][0]:.2f}%, {results['A_dense_permuted'][1]:.4f}")
        
        print("\n Pre-reset")
        model_a = evaluate_merged_models(permute_model_0, model_B_dense, config)
        train_accuracy, train_loss = evaluate(model_a, train_dl, config["device"])
        test_accuracy, test_loss = evaluate(model_a, test_dl, config["device"])
        print(
            f"(α=0.5): <-- Merged Model\tTrain Accuracy: {train_accuracy}\tTrain Loss: {train_loss:.4f}\t"
            f"Test Accuracy: {test_accuracy}\tTest Loss: {test_loss:.4f}"
        )
        
        print("\n Post-reset")
        reset_bn_stats(model_a, train_dl)
        results["Merged Model Post reset evaluation"] = evaluate(model_a, test_dl, config["device"])
        print(f"Merged Model Post reset evaluation: {results['Merged Model Post reset evaluation'][0]:.2f}%, {results['Merged Model Post reset evaluation'][1]:.4f}")

        print("\n Another reset (REPAIR)")
        repaired_model_a = repair_merged_model(permute_model_0, model_B_dense, model_a, train_dl, config, 0.5)
        results["Merged Model with REPAIR"] = evaluate(repaired_model_a, test_dl, config["device"])
        print(f"Merged Model with REPAIR: {results['Merged Model with REPAIR'][0]:.2f}%, {results['Merged Model with REPAIR'][1]:.4f}")
        
        ## Ensure this gives the same evaluation as model_A_sparse
        results["A_sparse_permuted"] = evaluate(
            permuted_model_A_sparse, test_dl, config["device"]
        )
        print(
            f"\n A_sparse_permuted: {results['A_sparse_permuted'][0]:.2f}%, {results['A_sparse_permuted'][1]:.4f}"
        )

        if compare_model_weights(original_model_A_sparse, permuted_model_A_sparse):
            print("\n\n The weights match for all layers.")
        else:
            print("\n\n The weights do not match for all layers.")  
    
    torch.save(permuted_model_A_sparse, config['save_path'] + 'permuted_model_a_sparse_sparsity_' + str(config['pruning']['sparsity']) + '_seed_' + str(config['seed']))


def train_naive_permuted(seed,k, config):

    results = {}

    current_time = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

    # Check selected backend is supported
    assert hasattr(torch.backends, config["device"]), f"Device {config['device']} not supported"
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Initialize wandb with a project name and a unique run name
    wandb.init(
        project=f"{config['project']}_compare",
        entity=config["entity"],
        name=f"rewind_{k}_{config['model_type']}; seed_{config['seed']}; {current_time}",
        config={'sparsity':config['pruning']['sparsity'],
                'rewind_point':k,
                'width':config['width_multiplier'],}
    )


    #decrease the sparse epochs based on rewind point
    adjusted_sparse_epochs = config["sparse_epochs"] - k
    train_dl, test_dl = cifar_dataloader(config["batch_size"], config)


    ################################################### Sparse training: LTH solution = W_A^{t=k} ⊙ M_A #############################################
    model_A_dense_init_for_LTH = get_model(config)
    load_path_rewound_point_A_for_LTH = f"{config['save_path']}Model_A_Dense_sparsity_{config['pruning']['sparsity']}_seed_{config['seed']}_epoch_{k}"
    model_A_dense_init_for_LTH.load_state_dict(torch.load(load_path_rewound_point_A_for_LTH))
    results["model_A_LTH_rewound_point_eval"] = evaluate(model_A_dense_init_for_LTH, test_dl, config["device"])
    print(f"model_A_LTH_rewound_point_eval: {results['model_A_LTH_rewound_point_eval'][0]:.2f}%, {results['model_A_LTH_rewound_point_eval'][1]:.4f}")

    original_model_A_sparse = torch.load(config['save_path']+'model_a_sparse_sparsity_' + str(config['pruning']['sparsity'])  +  '_seed_'+str(config['seed']))

    transfer_sparsity_resnet(original_model_A_sparse, model_A_dense_init_for_LTH)
    check_hooks(model_A_dense_init_for_LTH)
    print("Sparsity of the init after transfer sparsity: ",calculate_overall_sparsity_from_pth(model_A_dense_init_for_LTH))

    trainer(model_A_dense_init_for_LTH, config["optimizer"], adjusted_sparse_epochs, config["batch_size"], config["device"], "LTH", config, sparse=True, training_type="LTH")
    results["LTH"] = evaluate(
        model_A_dense_init_for_LTH, test_dl, config["device"]
    )
    print(f"LTH: {results['LTH'][0]:.2f}%, {results['LTH'][1]:.4f}")
    #################################################################################################################################################

    ################################################## Sparse training: Naive solution = W_B^{t=k} ⊙ M_A ############################################
    model_B_dense_init_naive = get_model(config)
    path_rewound_point_B_for_naive_and_permuted = f"{config['save_path']}Model_B_Dense_sparsity_{config['pruning']['sparsity']}_seed_{config['seed']}_epoch_{k}"
    model_B_dense_init_naive.load_state_dict(torch.load(path_rewound_point_B_for_naive_and_permuted))

    results["model_B_naive_rewound_point_eval"] = evaluate(model_B_dense_init_naive, test_dl, config["device"])
    print(f"model_B_naive_rewound_point_eval: {results['model_B_naive_rewound_point_eval'][0]:.2f}%, {results['model_B_naive_rewound_point_eval'][1]:.4f}")
    
    transfer_sparsity_resnet(original_model_A_sparse, model_B_dense_init_naive)
    check_hooks(model_B_dense_init_naive)
    print("Sparsity of the init after transfer sparsity: ",calculate_overall_sparsity_from_pth(model_B_dense_init_naive))
    
    trainer(model_B_dense_init_naive, config["optimizer"], adjusted_sparse_epochs, config["batch_size"], config["device"], "Model_B_Naive", config, sparse=True, training_type="naive")

    results["B_naive"] = evaluate(
        model_B_dense_init_naive, test_dl, config["device"]
    )
    print(f"B_naive: {results['B_naive'][0]:.2f}%, {results['B_naive'][1]:.4f}")
    ###############################################################################################################################################
    
    ################################################## Sparse training: Permuted solution = W_B^{k=i} ⊙ π(M_A) ####################################
    model_B_dense_init_perm = VGG11_nofc("VGG11", init_weights=config["init_weights"]).to(config["device"])
    model_B_dense_init_perm.load_state_dict(torch.load(load_path_rewound_point_B_k))
    ### Ensure this gives the same evaluation as the rewound point from above. ###
    results["model_b_init_perm_eval"] = evaluate(
        model_B_dense_init_perm, cifar_dataloader(config["batch_size"])[1], config["device"]
    )
    print(f"model_b_init_perm_eval: {results['model_b_init_perm_eval'][0]:.2f}%, {results['model_b_init_perm_eval'][1]:.4f}")

    ################################################# Sparse training: Permuted solution = W_B^{t=k} ⊙ π(M_A) ####################################
    model_B_dense_init_perm = get_model(config)
    model_B_dense_init_perm.load_state_dict(torch.load(path_rewound_point_B_for_naive_and_permuted))
    ## Ensure this gives the same evaluation as the rewound point from above. ###
    results["model_B_perm_rewound_point_eval"] = evaluate(
        model_B_dense_init_perm, test_dl, config["device"]
    )
    print(f"model_B_perm_rewound_point_eval: {results['model_B_perm_rewound_point_eval'][0]:.2f}%, {results['model_B_perm_rewound_point_eval'][1]:.4f}")

    permuted_model_A_sparse = torch.load(config['save_path'] + f"jan29_seed_{str(config['seed'])}_permuted_model_a_sparse_sparsity_rn20_c10_w1_" + str(config['pruning']['sparsity']))
    transfer_sparsity_resnet(permuted_model_A_sparse, model_B_dense_init_perm)
    check_hooks(model_B_dense_init_perm)
    print("Sparsity of the init after transfer sparsity: ",calculate_overall_sparsity_from_pth(model_B_dense_init_perm))
    
    trainer(model_B_dense_init_perm, config["optimizer"], adjusted_sparse_epochs, config["batch_size"], config["device"], "Model_B_Permuted", config, sparse=True, training_type="permuted")
    
    results["B_permuted"] = evaluate(
        model_B_dense_init_perm, test_dl, config["device"]
    )
    print(f"B_permuted: {results['B_permuted'][0]:.2f}%, {results['B_permuted'][1]:.4f}")
    ###############################################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str)
    parser.add_argument("-seed", type=int)
    parser.add_argument("-pretrain", type=str)
    parser.add_argument("--rewind",type=int)
    parser.add_argument("-slurm_tmpdir", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config['seed'] = args.seed

    config['save_path']= f"{config['save_path']}width_{config['width_multiplier']}/sparsity_{config['pruning']['sparsity']*100}/"
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'], exist_ok=True)
    print(config['save_path'])

    print(config)
    torch.manual_seed(args.seed)# set the random seed
    np.random.seed(args.seed)# set the random seed

    print(args.pretrain)
    if args.pretrain=='True':
        train_model_a_b(args.seed, config)
    elif  args.pretrain=='False':  
            train_naive_permuted(args.seed,args.rewind, config)
    else:
        raise ValueError('incorrect argument')
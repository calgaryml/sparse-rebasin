# Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry

This is the code repository for the paper "Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry", [accepted at ICML 2025](https://openreview.net/forum?id=BnfJSwtHLu), and available on arXiv at http://arxiv.org/abs/2505.05143.

## Citing our work

If you find this code useful, please cite our work with the following BibTex citation:

```bibtex
@inproceedings{mohammed2025sparsetraining,
  author = {Adnan, Mohammed and Jain, Rohan and Sharma, Ekansh and Krishnan, Rahul and Ioannou, Yani},
  title = {Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry},
  booktitle = {Forty-second International Conference on Machine Learning (ICML)},
  year = {2025},
  arxivid = {2505.05143},
  eprint = {2505.05143},
  eprinttype = {arXiv},
  venue = {{Vancouver, BC, Canada}},
  eventdate = {2025-07-13/2025-07-19},
}
```

Setup
-----

### Prerequisites

-   Python 3.x
-   PyTorch
-   Rest mentioned in `requirements.txt`
  
### Virtual Environment

We recommend running experiments/installing requirements in a python virtual environment:

1. Install venv module (if not already installed) as a user:

    ```
    python3 -m pip install --user virtualenv
    ```

2. Create a virtual environment named "env":

    ```
    python3 -m virtualenv env
    ```

3. Activate the virtual environment:

    ```
    source env/bin/activate
    ```

### Installation

1.  Clone the repository:

    ```
    git clone https://github.com/calgaryml/sparse-rebasin.git
    ```

3.  Install the required dependencies:

   
    ```
     pip install -r requirements.txt
    ```

## Usage

To run experiments for ResNet20 and VGG11 on CIFAR-10/100, setup your experiment and then simply execute in `experiment.py` script with Python, ensuring that the desired settings are specified in the `configs` and `configs_cifar100` folders for CIFAR-10 and CIFAR-100 respectively. 

To run the experiment after setup:

Part 1: Dense training, Pruning, and Permutation Matching.
```
python experiment.py -config <FILE_NAME> -seed <INT> -pretrain True
```
Part 2: Sparse training LTH, Naive, and the Permuted solutions.
```
python experiment.py -config <FILE_NAME> -seed <INT> -pretrain False --rewind <INT>
```

# sparse-rebasin

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

To run experiments, setup your experiment and then simply execute in `experiment.py` script with Python, ensuring that the desired settings are specified in the `config.yaml` file. 

To run the experiment after setup:

```
python experiment.py
```

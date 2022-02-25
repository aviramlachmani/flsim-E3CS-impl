# FLSim



## Installation

To install **FLSim**, all that needs to be done is clone this repository to the desired directory.

### Dependencies

**FLSim** uses [Anaconda](https://www.anaconda.com/distribution/) to manage Python and it's dependencies, listed in [`environment.yml`](environment.yml). To install the `fl-py37` Python environment, set up Anaconda (or Miniconda), then download the environment dependencies with:

```shell
conda env create -f environment.yml
```

## Usage

Before using the repository, make sure to activate the `fl-py37` environment with:

```shell
conda activate fl-py37
```


##  Running simulation for aviram and eden project:
1) choose database, we take as example cifar-10 database 
2) open configs/CIFAR-10/cifar-10.json
3) choose method value: `FedCs,random, pow-d, E3CS_0, E3CS_05, E3CS_08, E3CS_inc`
4) for run it on iid database write:
   `"IID": true'` . 
   else for non-iid write:
   `"IID": false,
    "bias": {
            "primary" : 0.8,
            "secondary" : 0.2
        }`
   
5) open models/CIFAR-10/fl_model.py and go to `get_optimizer()` :
   for FedAvg choose to return: `optim.SGD(...)`
   for FedProx choose to return: `FedProx(...)`
6) open server and go to name_file. write the current method that you use.
7) run form the commend-line in this folder :
`python run.py --config=configs/CIFAR-10/cifar-10.json --log=INFO`

if you wish to choose Emnist database do as 1-6 and run the commend:
   `python run.py --config=configs/MNIST/mnist.json --log=INFO`
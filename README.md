# redbnn

Reduced Bayesian Neural Networks are Deterministic Neural Networks with a Bayesian subset of weights. `redBNN` class computes a MAP estimate of the entire NN and then performs Bayesian inference on a chosen layer (`--reduction=layers`) or block (`--reduction=blocks`). 

This library is built upon `pyro` and `torchvision`: `redbnn` loads any pre-trained architecture from torchvision library and trains a deterministic Neural Network (`baseNN`) or a reduced Bayesian Neural Network (`redBNN`) using Stochastic Variational Inference (SVI) or Hamiltonian Monte Carlo (HMC) from pyro library. 

An example of training with `baseNN` or `redBNN` is provided by the script `training.py`.

## Install

Code runs with Python 3.7.6 on Ubuntu 18.10.

```
pip3 install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
cd src/
```
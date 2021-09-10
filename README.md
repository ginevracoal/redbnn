# redbnn

<div align="center">
	<img src="docs/_static/logo1.png" width="200">
</div>

Reduced Bayesian Neural Networks are Deterministic Neural Networks with a Bayesian subset of weights. `redBNN` class computes a MAP estimate of the entire NN and then performs Bayesian inference on a chosen layer (`--reduction=layers`) or block (`--reduction=blocks`). 

This library is built upon `pyro` and `torchvision`: `redbnn` loads any pre-trained architecture from torchvision library and trains a deterministic Neural Network (`baseNN`) or a reduced Bayesian Neural Network (`redBNN`) using Stochastic Variational Inference (SVI) or Hamiltonian Monte Carlo (HMC) from pyro library. 

An example of training with `baseNN` or `redBNN` is provided by the script `training.py`.

**Install**: `pip install redbnn`
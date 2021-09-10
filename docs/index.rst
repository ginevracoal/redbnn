.. redbnn documentation master file, created by
   sphinx-quickstart on Thu Sep  9 14:39:54 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to redbnn's documentation!
==================================

Reduced Bayesian Neural Networks are deterministic Neural Networks with a Bayesian subset of weights. `redBNN` class computes a MAP estimate of the entire NN and then performs Bayesian inference on a chosen layer (`--reduction=layers`) or block (`--reduction=blocks`).

This library is built upon pyro and torchvision: `redbnn` loads any pre-trained architecture from torchvision library and trains a deterministic Neural Network (`baseNN`) or a reduced Bayesian Neural Network (`redBNN`) using Stochastic Variational Inference (SVI) or Hamiltonian Monte Carlo (HMC) from pyro library.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   redbnn.bayesian_inference
   redbnn.nn
   redbnn.utils
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

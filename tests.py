import copy
import pyro
import torch
import warnings
import unittest
from pyro import poutine

from redbnn.nn.base import baseNN
from redbnn.nn.reduced import redBNN
from redbnn.utils.data import load_data
from redbnn.utils.pickling import load_from_pickle

dataloaders, input_size, num_classes = load_data(dataset_name='imagenette', data_dir='data/imagenette2-320',
                                                 subset_size=10)

filename = 'unittest'
architecture = 'resnet18'
savedir = 'data/'
device = 'cuda'


class TestredBNN(unittest.TestCase):

    # def test_base_training(self):

    #     model = baseNN(architecture='resnet18', input_size=input_size, num_classes=num_classes)
    #     params_to_update = model._initialize_model(feature_extract=True, use_pretrained=True)
    #     old_state_dict = model.network.state_dict()   

    #     model.train(dataloaders=dataloaders, num_iters=2, device=device)
    #     assert torch.all(torch.eq(list(old_state_dict.values())[0], list(model.network.state_dict().values())[0]))
    #     assert torch.all(~torch.eq(list(old_state_dict.values())[-1], list(model.network.state_dict().values())[-1]))

    # def test_save_load_baseNN(self):

    #     model = baseNN(architecture=architecture, input_size=input_size, num_classes=num_classes)
    #     model._initialize_model(feature_extract=True, use_pretrained=True)
    #     model.evaluate(dataloaders['test'], device=device)
    #     model.save(filename=filename, savedir='data/trained_models/unittest/')

    #     model._initialize_model(feature_extract=True, use_pretrained=True)
    #     model.load(filename=filename, savedir='data/trained_models/unittest/')
    #     model.evaluate(dataloaders['test'], device=device)

    def test_save_load_redBNN(self):

        for inference in ['svi', 'hmc']:

            model = redBNN(architecture=architecture, input_size=input_size, num_classes=num_classes, 
                        inference=inference, reduction='layers', bayesian_idx=4)        
            model._initialize_model()
            model.evaluate(dataloaders['test'], device=device, n_samples=2)
            model.save(filename='redBNN_'+str(inference), savedir='data/trained_models/unittest/')

            model = redBNN(architecture=architecture, input_size=input_size, num_classes=num_classes, 
                        inference=inference, reduction='layers', bayesian_idx=4) 
            model._initialize_model()
            model.load(filename='redBNN_'+str(inference), savedir='data/trained_models/unittest/')
            model.evaluate(dataloaders['test'], device=device, n_samples=2)

    # def test_posterior_samples_svi_blocks(self):
    #     warnings.filterwarnings('ignore')
    #     from pyro.distributions import Normal
    #     import torch.nn as nn
    #     import torch.nn.functional as nnf
    #     softplus = torch.nn.Softplus()

    #     network = redBNN(architecture=architecture, input_size=input_size, num_classes=num_classes, 
    #                 inference='svi', reduction='blocks', bayesian_idx=4)
    #     network.train(dataloaders=dataloaders, num_iters=2, device=device, eval_samples=3, svi_iters=2)

    #     old_state_dict = network.network.state_dict()

    #     network.to(device)

    #     network.network.eval()
    #     with torch.no_grad():

    #         inputs = next(iter(dataloaders['train']))[0]
    #         inputs = inputs.to(device)

    #         preds = []  
    #         for seed in [0,1,2]:
    #             pyro.set_rng_seed(seed)
    #             guide_trace = poutine.trace(network.guide).get_trace(inputs) 

    #             weights = {}
    #             for key, value in network.network.state_dict().items():
    #                 weights.update({str(key):value})

    #             for param_name, param in network.bayesian_weights.items():

    #                 dist = Normal(loc=guide_trace.nodes[param_name+"_loc"]["value"], 
    #                               scale=softplus(guide_trace.nodes[param_name+"_scale"]["value"]))
    #                 w = pyro.sample(param_name, dist)
    #                 weights.update({param_name:w})

    #             basenet_copy = copy.deepcopy(network.network)
    #             basenet_copy.load_state_dict(weights)
    #             preds.append(basenet_copy.forward(inputs))

    #             assert torch.all(torch.eq(old_state_dict['conv1.weight'], 
    #                              basenet_copy.state_dict()['conv1.weight']))
    #             assert torch.all(~torch.eq(old_state_dict['layer1.0.conv1.weight'], 
    #                              basenet_copy.state_dict()['layer1.0.conv1.weight']))

    # def test_posterior_samples_svi_layers(self):
    #     warnings.filterwarnings('ignore')
    #     from pyro.distributions import Normal
    #     import torch.nn as nn
    #     import torch.nn.functional as nnf
    #     softplus = torch.nn.Softplus()

    #     network = redBNN(architecture=architecture, input_size=input_size, num_classes=num_classes, 
    #                 inference='svi', reduction='layers', bayesian_idx=4)
    #     network.train(dataloaders=dataloaders, num_iters=2, device=device, eval_samples=3, svi_iters=2)

    #     old_state_dict = network.network.state_dict()

    #     network.to(device)

    #     network.network.eval()
    #     with torch.no_grad():

    #         inputs = next(iter(dataloaders['train']))[0]
    #         inputs = inputs.to(device)

    #         preds = []  
    #         for seed in [0,1,2]:
    #             pyro.set_rng_seed(seed)
    #             guide_trace = poutine.trace(network.guide).get_trace(inputs) 

    #             weights = {}
    #             for key, value in network.network.state_dict().items():
    #                 weights.update({str(key):value})

    #             for param_name, param in network.bayesian_weights.items():

    #                 dist = Normal(loc=guide_trace.nodes[param_name+"_loc"]["value"], 
    #                               scale=softplus(guide_trace.nodes[param_name+"_scale"]["value"]))
    #                 w = pyro.sample(param_name, dist)
    #                 weights.update({param_name:w})

    #             basenet_copy = copy.deepcopy(network.network)
    #             basenet_copy.load_state_dict(weights)
    #             preds.append(basenet_copy.forward(inputs))

    #             assert torch.all(torch.eq(old_state_dict['conv1.weight'], 
    #                              basenet_copy.state_dict()['conv1.weight']))
    #             assert torch.all(~torch.eq(old_state_dict['layer1.0.conv1.weight'], 
    #                              basenet_copy.state_dict()['layer1.0.conv1.weight']))

    # def test_posterior_samples_hmc_blocks(self):
    #     warnings.filterwarnings('ignore')
    #     from pyro.distributions import Normal
    #     import torch.nn as nn
    #     import torch.nn.functional as nnf
    #     softplus = torch.nn.Softplus()

    #     network = redBNN(architecture=architecture, input_size=input_size, num_classes=num_classes, 
    #                 inference='hmc', reduction='blocks', bayesian_idx=4)
    #     network.train(dataloaders=dataloaders, num_iters=2, device=device, eval_samples=3,  
    #                     hmc_samples=3, hmc_warmup=5)

    #     old_state_dict = network.network.state_dict()

    #     network.to(device)
    #     network.network.eval()
    #     with torch.no_grad():

    #         inputs = next(iter(dataloaders['train']))[0]
    #         inputs = inputs.to(device)

    #         preds = []  

    #         for seed in [0,1,2]:
    #             net = network.posterior_samples[seed]
    #             preds.append(net.forward(inputs))

    #             assert torch.all(torch.eq(old_state_dict['conv1.weight'], 
    #                              net.state_dict()['conv1.weight']))
    #             assert torch.all(~torch.eq(old_state_dict['layer1.0.conv1.weight'], 
    #                              net.state_dict()['layer1.0.conv1.weight']))

    # def test_posterior_samples_hmc_layers(self):
    #     warnings.filterwarnings('ignore')
    #     from pyro.distributions import Normal
    #     import torch.nn as nn
    #     import torch.nn.functional as nnf
    #     softplus = torch.nn.Softplus()

    #     network = redBNN(architecture=architecture, input_size=input_size, num_classes=num_classes, 
    #                 inference='hmc', reduction='layers', bayesian_idx=4)
    #     network.train(dataloaders=dataloaders, num_iters=2, device=device, eval_samples=3, 
    #                   hmc_samples=3, hmc_warmup=5)

    #     old_state_dict = network.network.state_dict()

    #     network.to(device)
    #     network.network.eval()
    #     with torch.no_grad():

    #         inputs = next(iter(dataloaders['train']))[0]
    #         inputs = inputs.to(device)

    #         preds = []  
    #         for seed in [0,1,2]:
    #             net = network.posterior_samples[seed]
    #             preds.append(net.forward(inputs))

    #             assert torch.all(torch.eq(old_state_dict['conv1.weight'], 
    #                              net.state_dict()['conv1.weight']))
    #             assert torch.all(~torch.eq(old_state_dict['layer1.0.conv1.weight'], 
    #                              net.state_dict()['layer1.0.conv1.weight']))

if __name__ == '__main__':
    unittest.main()
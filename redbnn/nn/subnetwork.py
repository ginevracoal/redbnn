import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro import poutine
from pyro.distributions import Normal, Categorical
from utils.networks import *
softplus = torch.nn.Softplus()

class SubNetwork(nn.Module):
    """ Subset of a Neural Network architecture, given start and end layers idxs.
    """

    def __init__(self, architecture):
        super(SubNetwork, self).__init__()
        """
        Args:
            architecture (str): Name of any torchvision architecture.            

        """
        self.architecture = architecture

    def initialize_model(self, original_model, reduction, start_layer_idx, end_layer_idx):
        """ Build subnetwork architecture from start_layer_idx to end_layer_idx (both idxs included) 
        of the original model. 

        Args:
            original_model (torchvision.models)
            reduction (str): Reduction method can be either `layers` or `blocks` depending on the desired structure.
            start_layer_idx (int): Index of the first layer in the subnetwork.
            end_layer_idx (int): Index of the last layer in the subnetwork.

        Returns:
            self
            
        """
        if hasattr(original_model, 'inference'):
            self.inference = original_model.inference

        if reduction == "layers":

            layer_idx = 0
            for block_name, block in list(original_model.network.named_children()):

                modules = []

                if len(list(block.children()))==0:
                    layer_idx+=1

                    if layer_idx>start_layer_idx and layer_idx<=end_layer_idx:
                        modules.append(block)
                        
                else:
                    for layer_name, layer in list(block.named_children()):
                        layer_idx+=1

                        if layer_idx>start_layer_idx and layer_idx<=end_layer_idx:
                            modules.append(layer)

                setattr(self, block_name, nn.Sequential(*modules))

        elif reduction == "blocks":

            blocks_dict = get_blocks_dict(original_model, mode="blocks", learnable_only=False)

            if start_layer_idx not in blocks_dict.keys():
                raise AttributeError("Wrong block idx.")

            for block_idx, block in blocks_dict.items():
                if block_idx >= start_layer_idx and block_idx < end_layer_idx:
                    setattr(self, block['name'], block['block'])
                else:
                    setattr(self, block['name'],  nn.Sequential())

        else:
            raise NotImplementedError

        return self

    def update_weights(self, bayesian_network, bayesian_input, sample_idx):
        pyro.set_rng_seed(sample_idx)

        if self.inference == "svi":
            guide_trace = poutine.trace(bayesian_network.guide).get_trace(bayesian_input) 

            weights = {}
            for param_name, param in self.state_dict().items():

                loc = guide_trace.nodes["module$$$"+param_name]["value"]
                scale = guide_trace.nodes["module$$$"+param_name]["value"]
                w = Normal(loc=loc, scale=softplus(scale)).sample()

                weights.update({str(param_name):w})

        else:
            raise NotImplementedError

        self.load_state_dict(weights)
        return self

    def forward(self, x, n_samples=None, sample_idxs=None, bayesian_network=None):

        n_samples = self.n_samples if hasattr(self, "n_samples") else n_samples
        sample_idxs = self.sample_idxs if hasattr(self, "sample_idxs") else sample_idxs

        if n_samples is not None:

            if sample_idxs is  None:
                sample_idxs = list(range(n_samples))

            out_list=[]
            for idx in sample_idxs:

                sampled_subnetwork = self.update_weights(bayesian_network=bayesian_network, 
                                                         bayesian_input=x, sample_idx=idx)
                out_list.append(sampled_subnetwork(x))

            out = torch.stack(out_list).mean(0)

        else:
            out = self._deterministic_forward(x)

        return out

    def _deterministic_forward(self, x):

        if self.architecture=="alexnet":
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            out = self.classifier(x)

        elif self.architecture=="resnet":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            out = self.fc(x)

        elif self.architecture=="vgg":
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            out = self.classifier(x)

        elif self.architecture=="densenet":
            x = self.features(x)
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            out = self.classifier(x)

        elif self.architecture=="wide_resnet":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            out = self.fc(x)

        else:
            raise NotImplementedError

        return out

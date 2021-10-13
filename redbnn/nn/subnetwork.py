import copy 
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pyro
# from pyro import poutine
# from pyro.distributions import Normal, Categorical
from redbnn.utils.networks import *
# softplus = torch.nn.Softplus()

from redbnn.nn.base import baseNN
import redbnn.bayesian_inference.svi as svi
import redbnn.bayesian_inference.hmc as hmc

from torchsummary import summary


class SubNetwork(baseNN):
    """ Subset of a Neural Network architecture, given start and end layers idxs.
    """

    def __init__(self, architecture, num_classes):
        super(SubNetwork, self).__init__(architecture=architecture, num_classes=num_classes)
        """
        Parameters:
            architecture (str): Name of any torchvision architecture.            

        """
        self.architecture = architecture
        self.num_classes = num_classes

    def initialize_model(self, reduction, start_idx, end_idx):
        """ Build subnetwork architecture from start_idx to end_idx (both idxs included) of the original model. 

        Parameters:
            original_model (torchvision.models)
            reduction (str): Reduction method can be either `layers` or `blocks` depending on the desired structure.
            start_idx (int): Index of the first layer/block in the subnetwork.
            end_idx (int): Index of the last layer/block in the subnetwork. 
                       
        """
        self._initialize_model(feature_extract=False, use_pretrained=True)

        if reduction == "layers":

            raise NotImplementedError

            # layer_idx = 0
            # for block_name, block in list(self.network.named_children()):

            #     modules = []

            #     if len(list(block.children()))==0:

            #         if layer_idx>=start_idx and layer_idx<=end_idx:
            #             modules.append(block)

            #         # print(layer_idx, start_idx, end_idx, len(list(block.children())), block)

            #         layer_idx+=1

            #     else:
            #         for _, subblock in list(block.named_children()):

            #             if layer_idx>=start_idx and layer_idx<=end_idx:
            #                 modules.append(subblock)

            #             # print(layer_idx, start_idx, end_idx, len(list(subblock.children())), subblock, )

            #             layer_idx+=1

            #     setattr(self.network, block_name, nn.Sequential(*modules))

        elif reduction == "blocks":

            blocks_dict = get_reduced_blocks_dict(self.network, learnable_only=False)

            if start_idx not in blocks_dict.keys():
                raise AttributeError("Wrong block idx.")

            for block_idx, block in blocks_dict.items():
                if block_idx >= start_idx and block_idx <= end_idx:
                    setattr(self.network, block['name'], block['block'])
                else:
                    setattr(self.network, block['name'], nn.Sequential())
                    # delattr(self.network, block['name'])

        else:
            raise NotImplementedError

        print(f"\n{reduction} {start_idx}-{end_idx} \n{self}\n")

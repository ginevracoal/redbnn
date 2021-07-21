import torch 
import numpy as np
from torch.utils.data import DataLoader


def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))

# def compute_activations(network, dataloader, layer_idx, device):
# 	activations = []
# 	for image, label in dataloader.dataset:
# 		image = image.unsqueeze(0).to(device)
# 		label = torch.tensor(label).unsqueeze(0).to(device)

# 		network.network.eval()
# 		activation = network.get_activation(x=image, layer_idx=layer_idx-1)
# 		activations.append(activation)
# 	activations = torch.cat(activations)
# 	return activations

# def compute_activations_dataloader(network, dataloader, layer_idx, device):
# 	activations = []
# 	labels = []
# 	for image, label in dataloader.dataset:
# 		image = image.unsqueeze(0).to(device)
# 		label = torch.tensor(label).unsqueeze(0).to(device)

# 		network.network.eval()
# 		activation = network.get_activation(x=image, layer_idx=layer_idx-1)

# 		activations.append(activation)
# 		labels.append(label)

# 	activations_dataloader = DataLoader(dataset=list(zip(torch.cat(activations), torch.cat(labels))), 
# 									 	batch_size=500, shuffle=False)
# 	return activations_dataloader

# def get_blocks_mode(model):

# 	if model.architecture in ["vgg", "alexnet"]:
# 		return "layers"

# 	elif model.architecture in ["resnet", "wide_resnet"]:
# 		return "blocks"


def get_blocks_dict(network, mode, learnable_only=True):

	# if mode is None:
	# 	mode = get_blocks_mode(model)

	def get_inner_blocks(name, block, layers_dict, layer_idx, learnable_layers_idxs):

		if len(list(block.children())) > 1:
			for inner_block_name, inner_block in list(block.named_children()):
				layer_idx, learnable_layers_idxs = get_inner_blocks(name=name+"."+inner_block_name, block=inner_block, 
																	layers_dict=layers_dict, layer_idx=layer_idx, 
																	learnable_layers_idxs=learnable_layers_idxs)
		else:
			layer = block

			if layer.__class__.__name__ in ["Conv2d", "Linear"]:
				learnable_layers_idxs.append(layer_idx)

			layers_dict[layer_idx] = {'name':name, 'layer':layer, 'category':layer.__class__.__name__}
			layer_idx += 1

		return layer_idx, learnable_layers_idxs

	blocks_dict = {}
	layers_dict = {}
	layer_idx = 0
	learnable_layers_idxs = []

	for block_name, block in list(network.named_children()):
		blocks_dict[layer_idx] = {'name':block_name, 'block':block}
		layer_idx, learnable_layers_idxs = get_inner_blocks(name=block_name, block=block, 
															layers_dict=layers_dict, layer_idx=layer_idx, 
															learnable_layers_idxs=learnable_layers_idxs)

	# get first layer in a block and its category
	for block_idx, block_val in blocks_dict.items():
		blocks_dict[block_idx].update({'category':layers_dict[block_idx]['category'],
									   'layer':layers_dict[block_idx]['layer']})

	# keep only learnable layers
	if learnable_only:
		sorted_idxs = sorted(list(set(learnable_layers_idxs) & set(layers_dict.keys())))
		layers_dict = {layer_idx: layers_dict[layer_idx] for layer_idx in sorted_idxs}

		sorted_idxs = sorted(list(set(learnable_layers_idxs) & set(blocks_dict.keys())))
		blocks_dict = {layer_idx: blocks_dict[layer_idx] for layer_idx in sorted_idxs}

	if mode=="layers":
		# print("\nLayers idxs:", list(layers_dict.keys()))
		return layers_dict

	elif mode=="blocks":
		# print("\nBlocks idxs:", list(blocks_dict.keys()))
		return blocks_dict

def get_first_layer(block): 

	def _inner_block(block, blocks_list):

		if len(list(block.children())) > 0:
			for inner_block in list(block.children()):
				blocks_list = _inner_block(inner_block, blocks_list)
		else:
			blocks_list.append(block)

		return blocks_list

	first_layer = _inner_block(block, blocks_list=[])[0]
	return first_layer
import torch 
import numpy as np
from torch.utils.data import DataLoader


def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))

def is_learnable(block):
	learnable_bool = 0

	if len(list(block.children())) > 1:

		for inner_block_name, inner_block in list(block.named_children()):
			inner_learnable_bool = is_learnable(inner_block)

			if inner_learnable_bool:
				learnable_bool = 1
	else:
		if block.__class__.__name__ in ["Conv2d", "Linear","BatchNorm2d"]:
			learnable_bool = 1

	return learnable_bool

def get_reduced_blocks_dict(network, learnable_only=True):

	blocks_dict = {}
	learnable_blocks_idxs = []

	for block_idx, (block_name, block) in enumerate(list(network.named_children())):
		is_block_learnable = is_learnable(block)
		blocks_dict[block_idx] = {'name':block_name, 'block':block, 'is_learnable':is_block_learnable}

		if is_block_learnable:
			learnable_blocks_idxs.append(block_idx)

	# keep only learnable layers
	if learnable_only:
		sorted_idxs = sorted(list(set(learnable_blocks_idxs) & set(blocks_dict.keys())))
		blocks_dict = {block_idx: blocks_dict[block_idx] for block_idx in sorted_idxs}

	return blocks_dict

def get_blocks_dict(network, mode, learnable_only=True):

	def get_inner_blocks(name, block, layers_dict, layer_idx, learnable_layers_idxs):

		if len(list(block.children())) > 1:
			for inner_block_name, inner_block in list(block.named_children()):
				layer_idx, learnable_layers_idxs = get_inner_blocks(name=name+"."+inner_block_name, block=inner_block, 
																	layers_dict=layers_dict, layer_idx=layer_idx, 
																	learnable_layers_idxs=learnable_layers_idxs)
		else:
			layer = block

			if layer.__class__.__name__ in ["Conv2d", "Linear","BatchNorm2d"]:
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
									   'layer':layers_dict[block_idx]['layer'],
									   'start_layer_idx':block_idx})

	# keep only learnable layers
	if learnable_only:
		sorted_idxs = sorted(list(set(learnable_layers_idxs) & set(layers_dict.keys())))
		layers_dict = {layer_idx: layers_dict[layer_idx] for layer_idx in sorted_idxs}

		sorted_idxs = sorted(list(set(learnable_layers_idxs) & set(blocks_dict.keys())))
		blocks_dict = {layer_idx: blocks_dict[layer_idx] for layer_idx in sorted_idxs}

	if mode=="layers":
		return layers_dict

	elif mode=="blocks":
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
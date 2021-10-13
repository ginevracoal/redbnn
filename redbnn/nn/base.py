import os
import time
import copy
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
from torchvision import models

from redbnn.utils.seeding import set_seed
from redbnn.utils.networks import get_blocks_dict

DEBUG=False

class baseNN(nn.Module):
    """ Deterministic Neural Network classifier. Used as the baseline model for reduced Bayesian 
    Neural Network (redBNN). """

    def __init__(self, architecture, num_classes):
        """
        Parameters:
            architecture (str): Name of any torchvision architecture.            
            num_classes (int): Number of classes in the classification problem.

        """
        super(baseNN, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes

    def _initialize_model(self, feature_extract, use_pretrained):
        """
        Loads a pretrained model from torchvision library and sets the parameters for training.
        Only works for models with a fully connected last layer, otherwise code needs to be adapted (e.g. special 
        case for squeezenet below).

        Parameters:
            feature_extract (bool): If True only updates the parameters with requires_grad=True, otherwise \
                                    updates all parameters.
            use_pretrained (bool): If True loads a pre-trained model from torchvision library with the chosen \
                                   architecture.

        Attributes:
            network (torchvision.model): Torchvision model with the chosen architecture.
            n_layers (int): Total number of layers in `self.network`.
            n_blocks (int): Total number of blocks (groups of layers) in `self.network`.

        Raises:
            NotImplementedError: If the architecture name is not found in torchvision library.

        Returns:
            (nn.Module.parameters generator): Model parameters to be updated during training.

        """
        architecture = self.architecture
        num_classes = self.num_classes

        if 'squeezenet' in architecture:
            network = getattr(models, architecture)(pretrained=use_pretrained)
            network = getattr(models, architecture)(pretrained=use_pretrained)
            self._set_parameter_requires_grad(network, feature_extract)
            network.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

        else:
            network = getattr(models, architecture)(pretrained=use_pretrained)
            self._set_parameter_requires_grad(network, feature_extract)

            layers = get_blocks_dict(network, mode='layers', learnable_only=False)
            last_layer = list(layers.values())[-1]
            num_ftrs = last_layer['layer'].in_features

            layer_module = last_layer['category']
            new_last_layer = getattr(nn, last_layer['category'])(num_ftrs, num_classes)

            split_layer_name = last_layer['name'].split('.')

            if len(split_layer_name)==1:
                setattr(network, last_layer['name'], new_last_layer)

            elif len(split_layer_name)==2:
                block_name, layer_index = split_layer_name
                setattr(eval('network.'+block_name), layer_index, new_last_layer)

            else:
                raise NotImplementedError

            print("\nLast layer:", new_last_layer)

        self.network = network
        self.n_layers = len(get_blocks_dict(network, mode='layers', learnable_only=False))
        self.n_blocks = len(get_blocks_dict(network, mode='blocks', learnable_only=False))
        params_to_update = self._set_params_updates(network, feature_extract)
        return params_to_update

    def _set_parameter_requires_grad(self, model, feature_extract):
        """ Sets requires_grad attribute of model parameters to False when we are feature extracting.
        
        Parameters:
            model (torchvision.models): Torchvision model. 
            feature_extract (bool): If True performs feature extraction and sets requires_grad=False for all \
                                    model parameters.

        """
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def _set_params_updates(self, model, feature_extract): 
        """ Set the parameters to be optimized during training. 

        Parameters:
            model (torchvision.models): Torchvision model.
            feature_extract (bool): If True only updates the parameters with `requires_grad=True`, otherwise \
                                    updates all parameters.

        """
        params_to_update = model.parameters()

        if DEBUG:
            print("\nParams to learn:")

        count = 0
        if feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    count += param.numel()
                    if DEBUG:
                        print("\t",name)
  
        else:
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    count += param.numel()
                    if DEBUG:
                        print("\t",name)
                
        print(f"\nTotal n. of params = {count}")    
        n_learnable_layers = len(get_blocks_dict(self, mode="layers", learnable_only=True))
        print(f"n. of layers = {self.n_layers} \nn. of learnable layers = {n_learnable_layers}")
        return params_to_update

    def train(self, dataloaders, device, num_iters, feature_extract=True, use_pretrained=True, is_inception=False):
        """ Trains self.network with stochastic gradient descent, using Adam optimizer.

        Parameters:
            dataloaders (dict): Dictionary containing training and validation torch dataloaders.
            device (str): Device chosen for training.
            num_iters (int): Number of training iterations.
            feature_extract (bool): If True only updates the parameters with `requires_grad=True`, otherwise \
                                    updates all parameters. Defaults to True.
            use_pretrained (bool): If True loads a pre-trained model from torchvision library with the chosen \
                                   architecture. Defaults to True.
            is_inception (bool): Special case for training torchvision inception network. Defaults to True.

        Returns: 
            (list): History of validation accuracies.

        """
        print("\n == baseNN training ==")

        device = torch.device(device)
        params_to_update = self._initialize_model(feature_extract=feature_extract, use_pretrained=use_pretrained)

        set_seed(0)
        model = self.network
        self.to(device)

        since = time.time()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params_to_update, lr=0.001)

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_iters):
            print('\nEpoch {}/{}'.format(epoch, num_iters - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  
                else:
                    model.eval() 

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):

                        if is_inception and phase == 'train':
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        preds = outputs.argmax(-1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        self.network.load_state_dict(best_model_wts)
        return val_acc_history

    def evaluate(self, dataloader, device, *args, **kwargs):
        """ Evaluate `self.network` on test data.

        Parameters:
            dataloader (torch.dataloader): Test dataloader.
            device (str): Device chosen for testing. 
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            (float): Test accuracy.

        """
        device = torch.device(device)
        self.to(device)

        self.network.eval()
        with torch.no_grad():

            correct_predictions = 0.0

            for x_batch, y_batch in dataloader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = self.forward(x_batch, *args, **kwargs)
                predictions = outputs.argmax(-1)
                correct_predictions += (predictions == y_batch).sum().item()
                
            accuracy = 100 * correct_predictions / len(dataloader.dataset)
            print("\nEval accuracy: %.2f%%" % (accuracy))
            return accuracy

    def get_activation(self, x, layer_idx):
        """ Get self.network activation corresponding to the chosen layer index.

        Parameters:
            x (torch.tensor): Input image.
            layer_idx (int): Index of the chosen layer.

        Returns:
            (torch.tensor): Input activation at the chosen layer.

        """      
        if layer_idx==-1:
            return x

        else:
            model = self.network
            model.eval() 
        
            activation={}

            def _activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook

            key = get_blocks_dict(model=self, learnable_only=False, mode="layers")[layer_idx]['name']

            for n, m in model.named_modules():
                if n==key:
                    m.register_forward_hook(_activation(n))

            with torch.no_grad():
                output = model(x)

            if DEBUG:
                print(layer_idx, key, " activation", activation[key].shape)

            return activation[key]

    def to(self, device):
        """ Sends `self.network` to the chosen device.

        Parameters:
            device (str): Name of the chosen device.

        """
        self.network = self.network.to(torch.device(device))

    def save(self, filename, savedir):
        """ Saves the learned parameters as torch.tensors on the CPU.

        Parameters:
            filename (str): Filename.
            savedir (str): Output directory.

        """
        filename += "_weights.pt"
        os.makedirs(savedir, exist_ok=True)
        self.to("cpu")
        torch.save(self.network.state_dict(), os.path.join(savedir, filename))
        print("\nSaving", os.path.join(savedir, filename))

    def load(self, filename, savedir):
        """ Loads the learned parameters.

        Parameters:
            filename (str): Filename.
            savedir (str): Output directory.

        """
        filename += "_weights.pt"

        self._initialize_model(feature_extract=False, use_pretrained=True)
        self.network.load_state_dict(torch.load(os.path.join(savedir, filename)))
        print("\nLoading", os.path.join(savedir, filename))

    def forward(self, inputs, softmax=False):
        """ Forward pass of the inputs through `self.network`. 

        Parameters:
            inputs (torch.tensor): Input images.
            softmax (bool): If True computes the softmax of each output tensor.

        Returns: 
            (torch.tensor): Output predictions.

        """
        preds = self.network.forward(inputs)
        return nnf.softmax(preds, dim=-1) if softmax else preds

    def zero_grad(self, *args, **kwargs):
        """ Set all gradients in `self.network` to zero.
        """
        return self.network.zero_grad()

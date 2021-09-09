import os

from redbnn.nn.base import baseNN
from redbnn.utils.seeding import set_seed
from redbnn.utils.networks import get_blocks_dict
import redbnn.bayesian_inference.svi as svi
import redbnn.bayesian_inference.hmc as hmc


class redBNN(baseNN):
    """ Reduced BNN is a Neural Network classifier with a single Bayesian block or a single Bayesian Layer, 
    depending on the chosen reduction method. 
    """

    def __init__(self, architecture, num_classes, inference, reduction, bayesian_idx):
        """
        Parameters:
            architecture (str): Name of any torchvision architecture.            
            num_classes (int): Number of classes in the classification problem.
            inference (str): Bayesian inference method.    
            reduction (str): Reduction method can be either `layers` or `blocks` depending on the desired structure.
            bayesian_idx (int): Index for the Bayesian layer or block in the architecture.
        """
        super(redBNN, self).__init__(architecture=architecture, num_classes=num_classes)
        self.inference = inference
        self.reduction = reduction
        self.bayesian_idx = bayesian_idx

    def _initialize_model(self):
        """ Loads a pre-trained model in to the architecture and sets the Bayesian parameters that need to be \
        inferred during training. 

        Attributes:
            bayesian_weights (dict): Bayesian parameters in the architecture.

        Raises:
            AttributeError: If the chosen `bayesian_idx` is not in the list of the allowed idxs. 

        """
        super(redBNN, self)._initialize_model(feature_extract=False, use_pretrained=True)
        allowed_idxs = list(get_blocks_dict(self.network, learnable_only=True, mode=self.reduction).keys())
       
        if self.bayesian_idx not in allowed_idxs:
            raise AttributeError(f"Chose bayesian_idx in: {allowed_idxs}")

        self.bayesian_weights = self._set_bayesian_weights(self.bayesian_idx)

    def _set_bayesian_weights(self, bayesian_idx):
        """ Builds the dictionary of Bayesian paramaters in the architecture. 

        Parameters:
            bayesian_idx (int): Index for the Bayesian layer or block in the architecture.
        
        Returns:
            Bayesian weights dictionary.

        """
        print("\nBayesian idx =", bayesian_idx)
        blocks_dict = get_blocks_dict(self.network, learnable_only=True, mode=self.reduction)

        if self.reduction=='blocks':
            subnet = 'block'

        elif self.reduction=='layers':
            subnet = 'layer'

        name = blocks_dict[bayesian_idx]['name']
        block_params_dict = {key:val for key, val in blocks_dict[bayesian_idx][subnet].named_parameters()}

        bayesian_weights_dict = {}
        for key, val in blocks_dict[bayesian_idx][subnet].named_parameters():
            bayesian_weights_dict[name+'.'+key] = val

        print("\nBayesian weights =", list(bayesian_weights_dict.keys()))
        return bayesian_weights_dict

    def train(self, dataloaders, device, use_pretrained=True, is_inception=False, num_iters=2, 
              svi_iters=10, hmc_samples=100, hmc_warmup=100, eval_samples=10):
        """ Freezes the deterministic parameters and infers the Bayesian paramaters using the chosen inference method.

        Parameters:
            dataloaders (dict): Dictionary containing training and validation torch dataloaders.
            device (str): Device chosen for training.
            use_pretrained (bool, optional): If True loads a pre-trained model from torchvision library with the \
                                             chosen architecture.
            is_inception (bool, optional): Special case for training torchvision inception network. 
            num_iters (int, optional): Number of iterations for fine-tuning the pre-trained network. 
            svi_iters (int, optional): Number of iterations for Stochastic Variational Inference.
            hmc_samples (int, optional): Number of Hamiltonian Monte Carlo samples.
            hmc_warmup (int, optional): Number of Hamiltonian Monte Carlo warmup samples.
            eval_samples (int, optional): Number of posterior samples drawn during the evaluation.

        Raises:
            NotImplementedError: If training is not implemented for the chosen inference method.

        """
        basenet = baseNN(architecture=self.architecture, num_classes=self.num_classes)
        self._initialize_model()

        basenet.train(dataloaders=dataloaders, num_iters=num_iters, feature_extract=True,
                     use_pretrained=use_pretrained, device=device)
        basenet._set_parameter_requires_grad(basenet.network, feature_extract=True)

        self.network = basenet.network

        ### new
        for key, param in self.network.named_parameters():
            if key in self.bayesian_weights.keys():
                param.requires_grad = True
            else:
                param.requires_grad = False
        ###########

        if self.inference=="svi":
            svi.train(redbnn=self, dataloaders=dataloaders,
                      device=device, num_iters=svi_iters, is_inception=is_inception)

        elif self.inference=="hmc":
            hmc.train(redbnn=self, dataloaders=dataloaders,
                      device=device, n_samples=hmc_samples, warmup=hmc_warmup, is_inception=is_inception)

        else:
            raise NotImplementedError

    def evaluate(self, dataloader, n_samples, device):
        """ Evaluate `self.network` on test data.

        Parameters:
            dataloader (torch.dataloader): Test dataloader.
            n_samples (int): Number of posterior samples drawn during the evaluation.
            device (str): Device chosen for testing. 

        Returns:
            (float): Test accuracy.

        """
        return super(redBNN, self).evaluate(dataloader=dataloader, device=device, n_samples=n_samples)

    def model(self, x_data, y_data):
        """ Stochastic function that implements the generative process and is conditioned on the observations. 

        Parameters:
            x_data (torch.tensor): Observed data points.
            y_data (torch.tensor): Labels of the observed data.

        """
        if self.inference=="svi":
            return svi.model(redbnn=self, x_data=x_data, y_data=y_data)

        elif self.inference=="hmc":
            return hmc.model(redbnn=self, x_data=x_data, y_data=y_data)

    def guide(self, x_data, y_data=None):
        """ Variational distribution for SVI inference method.

        Parameters:
            x_data (torch.tensor): Input data points.
            y_data (torch.tensor, optional): Labels of the input data.

        """
        if self.inference=="svi":
            return svi.guide(redbnn=self, x_data=x_data, y_data=y_data)

    def forward(self, inputs, n_samples=10, sample_idxs=None, expected_out=True, softmax=False):
        """ Forward pass of the inputs through the network using the chosen number of samples.

        Parameters:
            inputs (torch.tensor): Input images.
            n_samples (int, optional): Number of samples drawn during the evaluation.
            samples_idxs (list, optional): Random seeds used for drawing samples. If `samples_idxs` is None it is \
                                            defined as the range of integers from 0 to the maximum number of samples.
            expected_out (bool, optional): If True computes the expected output prediction, otherwise returns \
                                            all predictions as a `torch.tensor`. 
            softmax (bool, optional): If True computes the softmax of each output tensor.

        Returns: 
            (torch.Tensor): Output predictions

        """
        if sample_idxs is  None:
            sample_idxs = list(range(n_samples))
        else:
            if len(sample_idxs) != n_samples:
                raise ValueError("The number of sample idxs should match the number of posterior samples.")

        if self.inference=="svi":
            out = svi.forward(redbnn=self, inputs=inputs, n_samples=n_samples, sample_idxs=sample_idxs)

        elif self.inference=="hmc":
            out = hmc.forward(redbnn=self, inputs=inputs, n_samples=n_samples, sample_idxs=sample_idxs)
              
        preds = out.mean(0) if expected_out else out
        return nnf.softmax(preds, dim=-1) if softmax else preds

    def save(self, filename, savedir, hmc_samples=None):
        """ Saves the learned parameters as torch.tensors on the CPU.

        Parameters:
            filename (str): Filename.
            savedir (str): Output directory.
            hmc_samples (str, optional): Number of samples drawn during HMC inference, needed for saving models \
                                         trained with HMC.

        """
        self.to("cpu")
        
        if self.inference=="svi":
            svi.save(self, savedir=savedir, filename=filename)

        elif self.inference=="hmc":
            assert hmc_samples is not None
            hmc.save(self, savedir=savedir, filename=filename, hmc_samples=hmc_samples)

        print("\nSaving", os.path.join(savedir, filename))

    def load(self, filename, savedir, hmc_samples=None):
        """ Loads the learned parameters.

        Parameters:
            filename (str): Filename.
            savedir (str): Output directory.
            hmc_samples (str, optional): Number of samples drawn during HMC inference, needed for loading models \
                                         trained with HMC.

        """
        basenet = baseNN(architecture=self.architecture, num_classes=self.num_classes)
        self._initialize_model()

        if self.inference=="svi":
            svi.load(self, savedir=savedir, filename=filename)

        elif self.inference=="hmc":
            assert hmc_samples is not None
            hmc.load(self, savedir=savedir, filename=filename, hmc_samples=hmc_samples)

        print("\nLoading", os.path.join(savedir, filename))

    def to(self, device):
        """ Sends `self.network` to the chosen device.
        
        Parameters:
            device (str): Name of the chosen device.
        """
        self.network = self.network.to(device)



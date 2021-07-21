import os
from redbnn.nn.base import baseNN
from redbnn.utils.seeding import set_seed
from redbnn.utils.networks import get_blocks_dict
import redbnn.bayesian_inference.svi as svi
import redbnn.bayesian_inference.svi as hmc

class redBNN(baseNN):

    def __init__(self, architecture, input_size, num_classes, inference, reduction, bayesian_idx):

        super(redBNN, self).__init__(architecture=architecture, input_size=input_size, num_classes=num_classes)
        self.inference = inference
        self.reduction = reduction
        self.bayesian_idx = bayesian_idx

    def _initialize_model(self):
        """
        Load pretrained models, set parameters for training and specify last layer weights 
        as the only ones that need to be inferred.
        """
        super(redBNN, self)._initialize_model(feature_extract=False, use_pretrained=True)
        allowed_idxs = list(get_blocks_dict(self.network, learnable_only=True, mode=self.reduction).keys())
       
        if self.bayesian_idx not in allowed_idxs:
            raise AttributeError(f"Chose bayesian_idx in: {allowed_idxs}")

        self.bayesian_weights = self._set_bayesian_weights(self.bayesian_idx)

    def _set_bayesian_weights(self, bayesian_idx):

        print("\nBayesian layer idx =", bayesian_idx)
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
        """
        dataloaders: dictionary containing 'train' and 'val' dataloaders
        device: 'cpu' or 'cuda' 
        num_iters: number of training iterations for the baseNN
        svi_iters:
        hmc_samples:
        hmc_warmup:
        is_inception: flag for Inception v3 model
        """
        basenet = baseNN(architecture=self.architecture, input_size=self.input_size, num_classes=self.num_classes)
        bayesian_weights = self._initialize_model()

        basenet.train(dataloaders=dataloaders, num_iters=num_iters, feature_extract=True,
                     use_pretrained=use_pretrained, device=device)
        basenet._set_parameter_requires_grad(basenet.network, feature_extract=True)
        self.network = basenet.network

        # device = torch.device(device)
        if self.inference=="svi":
            svi.train(network=self, dataloaders=dataloaders, eval_samples=eval_samples,
                      device=device, num_iters=num_iters, is_inception=is_inception)

        elif self.inference=="hmc":
            hmc.train(network=self, dataloaders=dataloaders, eval_samples=eval_samples,
                      device=device, n_samples=n_samples, warmup=warmup, is_inception=is_inception)

        else:
            raise NotImplementedError

    def evaluate(self, dataloader, n_samples, device):
        return super(redBNN, self).evaluate(dataloader=dataloader, device=device, n_samples=n_samples)

    def model(self, x_data, y_data):

        if self.inference=="svi":
            return svi.model(network=self, x_data=x_data, y_data=y_data)

        elif self.inference=="hmc":
            return hmc.model(network=self, x_data=x_data, y_data=y_data)

        else:
            raise NotImplementedError

    def guide(self, x_data, y_data=None):

        if self.inference=="svi":
            return svi.guide(network=self, x_data=x_data, y_data=y_data)

    def forward(self, inputs, n_samples=5, sample_idxs=None, expected_out=True, softmax=False):

        if sample_idxs is  None:
            sample_idxs = list(range(n_samples))
        else:
            if len(sample_idxs) != n_samples:
                raise ValueError("The number of sample idxs should match the number of posterior samples.")

        if self.inference=="svi":
            out = svi.forward(network=self, inputs=inputs, n_samples=n_samples, sample_idxs=sample_idxs)

        elif self.inference=="hmc":
            out = hmc.forward(network=self, inputs=inputs, n_samples=n_samples, sample_idxs=sample_idxs)
              
        else:
            raise NotImplementedError

        preds = out.mean(0) if expected_out else out
        return nnf.softmax(preds, dim=-1) if softmax else preds

    def save(self, filename, savedir):
        self.to("cpu")

        filename += "_weights"
        
        if self.inference=="svi":
            svi.save(self, savedir, filename)

        elif self.inference=="hmc":
            hmc.save(self, savedir, filename)

        print("\nSaving", os.path.join(savedir, filename))

    def load(self, filename, savedir, *args, **kwargs):

        filename += "_weights"

        if self.inference=="svi":
            svi.load(bayesian_network=self, path=savedir, filename=filename)

        elif self.inference=="hmc":
            hmc.load(bayesian_network=self, path=savedir, filename=filename)

        print("\nLoading", os.path.join(savedir, filename))

    def to(self, device):
        """
        Send network to device.
        """
        self.network = self.network.to(device)
        # self.bayesian_layer = self.bayesian_layer.to(device)

        # if self.inference=="svi":
        #     svi.to(device)
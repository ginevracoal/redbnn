import os

from redbnn.nn.base import baseNN
from redbnn.utils.seeding import set_seed
from redbnn.utils.networks import get_blocks_dict, get_reduced_blocks_dict
import redbnn.bayesian_inference.svi_subnetwork as svi
import redbnn.bayesian_inference.hmc as hmc
from redbnn.nn.subnetwork import SubNetwork

from torchsummary import summary


class redBNN(baseNN):


    def __init__(self, architecture, num_classes, inference, reduction, bayesian_idx):

        if reduction=='layers':
            raise NotImplementedError
        else:
            super(redBNN, self).__init__(architecture=architecture, num_classes=num_classes)
            self.inference = inference
            self.reduction = reduction
            self.bayesian_idx = bayesian_idx

    def _initialize_model(self, model):

        self.network = model

        # super(redBNN, self)._initialize_model(feature_extract=False, use_pretrained=True)
        allowed_idxs = list(get_blocks_dict(self.network, learnable_only=True, mode=self.reduction).keys())
       
        if self.bayesian_idx not in allowed_idxs:
            raise AttributeError(f"Chose bayesian_idx in: {allowed_idxs}")

        # self.bayesian_weights = self._set_bayesian_weights(self.bayesian_idx)
        self._set_bayesian_subnetwork(architecture=self.architecture, reduction=self.reduction,
                                        bayesian_idx=self.bayesian_idx)
        self._set_parameter_requires_grad()


    def _set_bayesian_subnetwork(self, architecture, reduction, bayesian_idx):

        print("\nBayesian idx =", bayesian_idx)#, "\nReduction method =", reduction)

        total_n_layers = len(get_blocks_dict(self.network, mode='layers', learnable_only=False))

        subnet='block'
        blocks_dict = get_reduced_blocks_dict(self.network, learnable_only=True)
        name = blocks_dict[bayesian_idx]['name']
        bayesian_weights_dict = {}
        for key, val in blocks_dict[bayesian_idx][subnet].named_parameters():
            bayesian_weights_dict[name+'.'+key] = val
        self.bayesian_weights = bayesian_weights_dict
        print("Bayesian weights =", list(bayesian_weights_dict.keys()))

        print("\n--- Activation subnetwork ---")
        activation_subnetwork = SubNetwork(architecture=architecture, num_classes=self.num_classes)
        activation_subnetwork.initialize_model(reduction=reduction, start_idx=0, end_idx=bayesian_idx-1)
        self.activation_subnetwork = activation_subnetwork

        print("\n--- Bayesian subnetwork ---")
        bayesian_subnetwork = SubNetwork(architecture=architecture, num_classes=self.num_classes)
        bayesian_subnetwork.initialize_model(reduction=reduction, start_idx=bayesian_idx, end_idx=bayesian_idx)
        self.bayesian_subnetwork = bayesian_subnetwork

        print("\n--- Output subnetwork ---")
        end_idx = len(get_reduced_blocks_dict(self.network, learnable_only=False))
        output_subnetwork = SubNetwork(architecture=architecture, num_classes=self.num_classes)
        output_subnetwork.initialize_model(reduction=reduction, start_idx=bayesian_idx+1, end_idx=end_idx)
        self.output_subnetwork = output_subnetwork

    def _set_parameter_requires_grad(self):
        """ Sets requires_grad attribute of model parameters to False for weights in the determinstic base network and \
        to True only for bayesian weights in self.bayesian_subnetwork.
        """        
        for key, param in self.network.named_parameters():
            param.requires_grad = False

        for key, param in self.bayesian_subnetwork.named_parameters():
            if key in self.bayesian_weights.keys():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
    def train(self, dataloaders, device, use_pretrained=True, is_inception=False, num_iters=2, 
              svi_iters=10, hmc_samples=100, hmc_warmup=100, eval_samples=10):

        basenet = baseNN(architecture=self.architecture, num_classes=self.num_classes)
        basenet.train(dataloaders=dataloaders, num_iters=num_iters, feature_extract=True,
                     use_pretrained=use_pretrained, device=device)
        self._initialize_model(model=basenet.network)

        if self.inference=="svi":
            svi.train(nn=self, 
                # activation_subnetwork=self.activation_subnetwork, output_subnetwork=self.output_subnetwork,
                    model=self.model, guide=self.guide, dataloaders=dataloaders, 
                      device=device, num_iters=svi_iters, is_inception=is_inception)

        elif self.inference=="hmc":
            hmc.train(nn=self, dataloaders=dataloaders,
                      device=device, n_samples=hmc_samples, warmup=hmc_warmup, is_inception=is_inception)

        else:
            raise NotImplementedError

    def evaluate(self, dataloader, n_samples, device):

        return super(redBNN, self).evaluate(dataloader=dataloader, device=device, n_samples=n_samples)

    def model(self, x_data, y_data):

        if self.inference=="svi":
            return svi.model(nn=self, x_data=x_data, y_data=y_data)#,
                             # activation_subnetwork=self.activation_subnetwork, output_subnetwork=self.output_subnetwork)

        elif self.inference=="hmc":
            return hmc.model(nn=self, x_data=x_data, y_data=y_data)#,
                             # activation_subnetwork=self.activation_subnetwork, output_subnetwork=self.output_subnetwork)

    def guide(self, x_data, y_data=None):
  
        if self.inference=="svi":
            return svi.guide(nn=self, 
                            # activation_subnetwork=self.activation_subnetwork, output_subnetwork=self.output_subnetwork,
                            x_data=x_data, y_data=y_data)

    def forward(self, inputs, n_samples=10, sample_idxs=None, expected_out=True, softmax=False):


        # bay_inputs = self.activation_subnetwork(inputs)

        if self.inference=="svi":
            out = svi.forward(network=self.bayesian_subnetwork, inputs=inputs, 
                                # activation_subnetwork=self.activation_subnetwork, output_subnetwork=self.output_subnetwork,
                                    n_samples=n_samples, sample_idxs=sample_idxs)

        elif self.inference=="hmc":
            out = hmc.forward(network=self.bayesian_subnetwork, inputs=inputs, 
                                    n_samples=n_samples, sample_idxs=sample_idxs)
              
        # out = self.output_subnetwork(bay_out)

        preds = out.mean(0) if expected_out else out
        return nnf.softmax(preds, dim=-1) if softmax else preds

    def save(self, filename, savedir, hmc_samples=None):

        self.to("cpu")
        # torch.save(self.activation_subnetwork.state_dict(), os.path.join(savedir, filename+"_activation_subnet_weights.pt"))

        if self.inference=="svi":
            svi.save(self, savedir=savedir, filename=filename)

        elif self.inference=="hmc":
            assert hmc_samples is not None
            hmc.save(self, savedir=savedir, filename=filename, hmc_samples=hmc_samples)

        print("\nSaving", os.path.join(savedir, filename))

    def load(self, filename, savedir, hmc_samples=None):

        basenet = baseNN(architecture=self.architecture, num_classes=self.num_classes)
        self._initialize_model()
        
        # self.activation_subnetwork = torch.load(os.path.join(savedir, filename+"_activation_subnet_weights.pt"))

        if self.inference=="svi":
            svi.load(self, savedir=savedir, filename=filename)

        elif self.inference=="hmc":
            assert hmc_samples is not None
            hmc.load(self, savedir=savedir, filename=filename, hmc_samples=hmc_samples)

        print("\nLoading", os.path.join(savedir, filename))

    def to(self, device):

        self.network = self.network.to(device)
        self.activation_subnetwork = self.activation_subnetwork.to(device)
        self.bayesian_subnetwork = self.bayesian_subnetwork.to(device)
        self.output_subnetwork = self.output_subnetwork.to(device)



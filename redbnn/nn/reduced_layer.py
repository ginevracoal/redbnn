from utils.data import *
from utils.paths import *
from networks.torchvision_baseNN import *
from utils.networks import *
from torchsummary import summary

from bayesian_inference.pyro_svi_single_layer import Model
import bayesian_inference.pyro_svi_single_layer as svi
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive


class redBNN_layer(baseNN):

    def __init__(self, baseNN, inference, use_pretrained=True):

        super(redBNN, self).__init__(architecture=baseNN.architecture, dataset_name=baseNN.dataset_name, 
                                     num_classes=baseNN.num_classes, feature_extract=False, use_pretrained=use_pretrained)
        
        self.inference = inference
        self.n_layers = baseNN.n_layers
        self.name = str(self.architecture)+"_"+str(self.inference)+"_"+str(self.dataset_name)+"_redBNN"

    def initialize_model(self, baseNN, bayesian_layer_idx):
        """
        Load pretrained models, set parameters for training and specify last layer weights 
        as the only ones that need to be inferred.
        """
        self.network = baseNN.network 

        if bayesian_layer_idx < 0 or bayesian_layer_idx > self.n_layers:
            raise AttributeError("Pass 0 <= bayesian_layer_idx <= n. layers")

        self.bayesian_layer_idx = bayesian_layer_idx
        self.bayesian_layer_dict = self._set_bayesian_layer_dict(bayesian_layer_idx)
        self.bayesian_layer = self._set_bayesian_layer(bayesian_layer_idx)
        self.name += "_bayLayerIdx="+str(bayesian_layer_idx)

    def _set_bayesian_layer_dict(self, bayesian_layer_idx):

        layers_dict = get_blocks_dict(model=self, learnable_only=True, mode="layers")
        blocks_dict_keys = get_blocks_dict(model=self, learnable_only=True, mode="blocks").keys()
        # print("\n Bayesian layers idxs =", list(blocks_dict_keys))

        if bayesian_layer_idx not in blocks_dict_keys:
            raise ValueError(f"Choose idx from: {blocks_dict_keys}")

        else:
            print("\nBayesian layer idx =", bayesian_layer_idx)

            layer_name = layers_dict[bayesian_layer_idx]['name']
            named_params_dict = {key:val for key, val in self.network.named_parameters()}

            bayesian_layer_dict = {}
            bayesian_layer_dict[layer_name+'.weight'] = named_params_dict[layer_name+'.weight']
            if layer_name+'.bias' in named_params_dict.keys():
                bayesian_layer_dict[layer_name+'.bias'] = named_params_dict[layer_name+'.bias']

            return bayesian_layer_dict

    def _set_bayesian_layer(self, bayesian_layer_idx):

        blocks_dict = get_blocks_dict(model=self, learnable_only=True, mode="blocks")
        bayesian_block = blocks_dict[bayesian_layer_idx]['block']
        bayesian_layer = get_first_layer(bayesian_block)
        print("\nBayesian layer =", bayesian_layer)
        return nn.Sequential(bayesian_layer)

    def train(self, dataloaders, device, eval_samples, is_inception=False, num_iters=None, n_samples=None, warmup=None):
        """
        dataloaders: dictionary containing 'train', 'test' and 'val' dataloaders
        device: "cpu" or "cuda" device 
        num_iters: number of training iterations
        is_inception: flag for Inception v3 model
        """
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
            # return svi.model(network=self, x_data=x_data, y_data=y_data)
            return Model(self)

        elif self.inference=="hmc":
            return hmc.model(network=self, x_data=x_data, y_data=y_data)

        else:
            raise NotImplementedError

    def guide(self, x_data, y_data=None):

        if self.inference=="svi":
            # return svi.guide(network=self, x_data=x_data, y_data=y_data)
            return AutoDiagonalNormal(self.model)

    def forward(self, inputs, n_samples=3, sample_idxs=None, expected_out=True, softmax=False):

        # change external attack libraries behavior #
        n_samples = self.n_samples if hasattr(self, "n_samples") else n_samples
        sample_idxs = self.sample_idxs if hasattr(self, "sample_idxs") else sample_idxs
        expected_out = self.expected_out if hasattr(self, "expected_out") else expected_out
        #############################################

        if hasattr(self, 'n_samples'):
            n_samples = self.n_samples
        else:
            if n_samples is None:
                raise ValueError("Set the number of posterior samples.")

        if sample_idxs is  None:
            sample_idxs = list(range(n_samples))
        else:
            if len(sample_idxs) != n_samples:
                raise ValueError("Number of sample_idxs should match number of samples.")

        if self.inference=="svi":
            out = svi.forward(network=self, inputs=inputs, n_samples=n_samples, sample_idxs=sample_idxs)

        elif self.inference=="hmc":
            out = hmc.forward(network=self, inputs=inputs, n_samples=n_samples, sample_idxs=sample_idxs)
              
        else:
            raise NotImplementedError

        preds = out.mean(0) if expected_out else out
        return nnf.softmax(preds, dim=-1) if softmax else preds

    def save(self, savedir):
        self.to("cpu")

        filename=self.name+"_weights"
        
        if self.inference=="svi":
            svi.save(self, savedir, filename)

        elif self.inference=="hmc":
            hmc.save(self, savedir, filename)

        print("\nSaving", os.path.join(savedir, filename))

    def load(self, savedir, device, *args, **kwargs):

        filename=self.name+"_weights"

        if self.inference=="svi":
            svi.load(bayesian_network=self, path=savedir, filename=filename, *args, **kwargs)

        elif self.inference=="hmc":
            hmc.load(bayesian_network=self, path=savedir, filename=filename, *args, **kwargs)

        self.to(device)
        print("\nLoading", os.path.join(savedir, filename))

    def to(self, device):
        """
        Send network to device.
        """
        self.network = self.network.to(device)
        self.bayesian_layer = self.bayesian_layer.to(device)

        if self.inference=="svi":
            svi.to(device)
from netdissect.modelconfig import create_instrumented_model
from types import SimpleNamespace
import numpy as np

class SGSampler():
    def __init__(self, model, output_class, layers, components_dir, device,  **kwargs):
        super().__init__()
        self.model = model
        self.output_class = output_class
        self.layers = layers if isinstance(layers, list) else [layers]
        self.device = device
        self.kwargs = kwargs
        self.components = None
        self.latent_dirs = None
        self.latent_stdevs = None
        self.latent_mean = None
        self.load_components(components_dir)

    def load_components(self, path_to_components):
        # Load components and stdevs from a numpy file
        comps = np.load(path_to_components)
        self.latent_dirs = comps['lat_comp']
        self.latent_stdevs = comps['lat_stdev']
        self.latent_mean = comps['lat_mean']


    def load_model(self):
        self.model.eval()

        inst = self.kwargs.get('inst', None)
        if self.inst:
            self.inst.close()

        module_names = [name for (name, _) in self.model.named_modules()]
        for layer_name in self.layers:
            if layer_name not in module_names:
                raise RuntimeError(f"Unknown layer '{layer_name}'")

        if hasattr(self.model, 'use_z'):
            self.model.use_z()

        self.inst = create_instrumented_model(SimpleNamespace(
            model=self.model,
            layers=self.layers,
            cuda=self.device.type == 'cuda',
            gen=True,
            latent_shape=self.model.get_latent_shape()
        ))

        if self.kwargs.get('use_w', False):
            self.model.use_w()
        
        return self.inst

    def sample_pca_space(self, n_samples):
        random_samples = np.random.randn(n_samples, self.latent_dirs.shape[0]) * self.latent_stdevs
        return random_samples

    def reconstruct_from_pca(self, pca_samples):
        original_data = np.dot(pca_samples, self.latent_dirs[:,0,:]) + self.latent_mean
        return original_data

    def project_to_pca(self, new_data):
        centered_data = new_data - self.latent_mean
        pca_projection = np.dot(centered_data, self.latent_dirs)
        return pca_projection

    def sample_latent(self, seed, truncation, num, scale, start_layer, end_layer):
        np.random.seed(seed)
        w = self.model.sample_latent(1, seed=seed).cpu().numpy()
        self.model.truncation = truncation
        direction = self.latent_dirs[num]
        
        w = [w] * self.model.get_max_latents()  # Assume this method is defined
        for l in range(start_layer, end_layer):
            w[l] = w[l] + direction * scale

        return w, self.model.sample_np(w) 




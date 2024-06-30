import torch
import numpy as np
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from pytorch_lightning import LightningModule
from torchvision.models import vgg19
import torch.nn as nn
from src.modules.metrics import load_metrics
from typing import Any, List
from omegaconf import DictConfig
import hydra
from src.utils.utils import getcls
from torch.utils.data import DataLoader, TensorDataset


class BayesianOptimization(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Model loop (model_step)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: DictConfig,
        sampler: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LightningModule with standalone train, val and test dataloaders.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__()
        self.model = getcls(model.model._target_)
        self.mll = getcls(model.mll._target_)
        self.acqf = getcls(model.acqf._target_)
        self.acqf_args = {key: value for key, value in model.acqf.items() if key != '_target_'}
        self.fit = getcls(model.fit._target_)
        self.sampler = hydra.utils.instantiate(sampler)
        main_metric = load_metrics(
            model.objective
        )
        self.train_metric = main_metric.clone()
        self.save_hyperparameters(logger=False)
        self.target = None 
        self.train_x = None

    def update_gp_model(self):
        self.model = self.model(self.train_x, self.train_y)
        self.mll = self.mll(self.model.likelihood, self.model)
        self.fit(self.mll)

    def on_train_start(self) -> None:
        self.train_metric.reset()
        self.target = self.sampler.model.sample_latent(1).cpu().detach().numpy() 
        self.img_target = self.sampler.model.sample_np(self.target)
        self.train_x = torch.tensor(self.sampler.sample_pca_space(1)).float()
        rec_x = self.sampler.reconstruct_from_pca(self.train_x).astype(np.float32)
        self.img_x = self.sampler.model.sample_np(rec_x)
        self.train_y = torch.tensor([self.train_metric(self.img_x,self.img_target)]).unsqueeze(-1).float()
                # Define the bounds of your optimization based on PCA variances (assuming normality and using some scale factor)
        self.bounds = torch.stack([
            torch.tensor([-1 * stdev for stdev in self.sampler.latent_stdevs]),  # Lower bounds
            torch.tensor([1 * stdev for stdev in self.sampler.latent_stdevs])   # Upper bounds
        ]).double()

        self.update_gp_model()



    def training_step(self, batch: Any, batch_idx: int) -> Any:
        print('train')

        with torch.enable_grad():

            candidates, _ = optimize_acqf(
                acq_function=self.acqf(self.model, **self.acqf_args),
                bounds=self.bounds,
                q=1,  # One new candidate point
                num_restarts=5,
                raw_samples=20  # Initial random points
            )
            rec_x = self.sampler.reconstruct_from_pca(candidates.detach().numpy()).astype(np.float32)
            self.img_x = self.sampler.model.sample_np(rec_x)

            # Evaluate new candidate point
            new_y = torch.tensor([self.train_metric(self.img_x,self.img_target)])
            
            # Update training data
            self.train_x = torch.cat([self.train_x, candidates]).detach()
            self.train_y = torch.cat([self.train_y, new_y.unsqueeze(0)]).detach()

            self.update_gp_model()

        self.log(
            f"{self.train_metric.__class__.__name__}/train",
            new_y,
            **self.logging_params,
        )

        # Lightning keeps track of `training_step` outputs and metrics on GPU for
        # optimization purposes. This works well for medium size datasets, but
        # becomes an issue with larger ones. It might show up as a CPU memory leak
        # during training step. Keep it in mind.
        return 
    
    def configure_optimizers(self):
        return None  # We don't use an optimizer for this task





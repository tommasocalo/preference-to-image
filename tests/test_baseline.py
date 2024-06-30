import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from hydra.utils import instantiate
from tests.helpers.run_if import RunIf
from typing import Any, Dict
import numpy as np 
import hydra
import torch

import warnings

def test_baseline_model(cfg_baseline):
    """Run for 1 train, val and test step."""
    warnings.filterwarnings('ignore')
    HydraConfig().set_config(cfg_baseline)
    with open_dict(cfg_baseline):
      sampler = instantiate(cfg_baseline.module.sampler)
      objective = instantiate(cfg_baseline.module.model.objective)

      sample_x = torch.tensor(sampler.sample_pca_space(1)).float()
      train_x = sampler.reconstruct_from_pca(sample_x).astype(np.float32)
      img_x = sampler.model.sample_np(train_x)
      target = sampler.model.sample_latent(1).cpu().detach().numpy() 
      img_target = sampler.model.sample_np(target)
      train_y = torch.tensor([objective(img_x,img_target)]).unsqueeze(-1).float()

      model = instantiate(cfg_baseline.module.model.model,sample_x,train_y)
      mll = instantiate(cfg_baseline.module.model.mll, model.likelihood, model)
      acqf = instantiate(cfg_baseline.module.model.acqf, model)

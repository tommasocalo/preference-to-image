from typing import Any, Dict

import omegaconf
import pytest

from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from PIL import Image
import numpy as np
from src.modules.metrics import load_metrics


_TEST_METRICS_CFG = (
    {"_target_": "src.modules.metrics.FaceNetSimilarity", 
     "facenet": {"_target_": "src.modules.metrics.models.FaceNetModel"}},
)


@pytest.mark.parametrize("metric_cfg", _TEST_METRICS_CFG)
def test_metric_cfg(metric_cfg: Dict[str, Any]):
    metrics_cfg = {
        "main": metric_cfg,
    }
    cfg = omegaconf.OmegaConf.create(metrics_cfg)
    _ = load_metrics(cfg)


@pytest.mark.parametrize("metric_cfg", _TEST_METRICS_CFG)
def test_face_similarity(metric_cfg: Dict[str, Any]):
    metrics_cfg = {
        "main": metric_cfg,
    }
    cfg = omegaconf.OmegaConf.create(metrics_cfg)

    face_similarity = load_metrics(cfg)

    # Dummy image generation
    transform = Compose([
        Resize((299, 299)),  # Resize to the input size of InceptionResnetV1
        ToTensor(),  # Convert to tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like ImageNet
    ])

    # Create two dummy images (random noise)
    image1 = Image.fromarray(np.uint8(np.random.rand(299, 299, 3) * 255))
    image2 = Image.fromarray(np.uint8(np.random.rand(299, 299, 3) * 255))

    # Apply transformations
    tensor1 = transform(image1)
    tensor2 = transform(image2)

    # Add batch dimension
    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)
    # Calculate cosine similarity
    cosine_similarity = face_similarity(tensor1, tensor1)

    print("Cosine Similarity:", cosine_similarity.item())

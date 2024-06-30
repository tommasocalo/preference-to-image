from typing import Optional
import torch
from pytorch_lightning.utilities import FLOAT32_EPSILON
from torchmetrics import Metric
import torch
from torchmetrics import Metric
from PIL import Image
import numpy as np

class FaceNetSimilarity(Metric):
    def __init__(self, facenet, distance_function=torch.nn.functional.cosine_similarity, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.facenet = facenet.eval()
        self.distance_function = distance_function
        self.add_state("similarity", default=torch.tensor(0.), dist_reduce_fx="mean")

    def forward(self, images1, images2):
        # Generate embeddings for both image batches
        target_image = Image.fromarray((images2 * 255).astype(np.uint8))
        target_resized_image = target_image.resize((160, 160), Image.NEAREST)
        target_tensor = torch.from_numpy(np.array(target_resized_image)).unsqueeze(0).permute(0, 3, 1, 2)
        target_embedding = self.facenet(target_tensor.float())

        X_image = Image.fromarray((images1 * 255).astype(np.uint8))
        X_resized_image = X_image.resize((160, 160), Image.NEAREST)
        X_tensor = torch.from_numpy(np.array(X_resized_image)).unsqueeze(0).permute(0, 3, 1, 2)
        X_embedding = self.facenet(X_tensor.float())

        # Compute similarity using the specified distance function
        similarity = self.distance_function(target_embedding, X_embedding, dim=1).mean()

        # Update state with new similarity value
        self.update(similarity)
        return self.similarity

    def update(self, similarity):
        # Update the running similarity value
        self.similarity = similarity

    def compute(self):
        # Return the average of the similarity calculated from all updates
        return self.similarity
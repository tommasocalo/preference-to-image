import torch
import torch.nn as nn
from torchvision.models import resnet34
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceNetModel(nn.Module):
    def __init__(self, embedding_dimension=128, pretrained=True):
        super(FaceNetModel, self).__init__()
        # Load a pre-trained ResNet-34 model
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        # Replace the final fully connected layer


    def forward(self, x):
        # Forward pass through the network
        embeddings = self.resnet(x)

        return embeddings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class ImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super(ImageEncoder, self).__init__()
        self.out_dim = out_dim
        self.vgg = vgg19(pretrained=True)
        self.vgg.avgpool = nn.Sequential()
        self.vgg.classifier = nn.Sequential()
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        """Converts an image into a feature vector
        
        Args:
            x (N, Channels, Height, Width): Due to vgg, fixed to (3, 448, 448)
        
        Returns:
            feats (N, 14 * 14, out_dim): Feature representation of image. 14 * 14 is fixed due to using of vgg
        """
        feats = self.vgg(x)
        batch_size = x.shape[0]
        feats = feats.view(batch_size, 14 * 14, 512)
        return self.fc(feats)


if __name__ == '__main__':
    model = ImageEncoder(out_dim=250)
    inp = torch.rand((1, 3, 448, 448))
    out = model(inp)
    assert tuple(out.shape) == (1, 14 * 14, 250)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, resnet50
from torchinfo import summary


class ImageEncoder(nn.Module):
    def __init__(self, out_dim, trainable=False):
        super(ImageEncoder, self).__init__()
        self.out_dim = out_dim
        # self.vgg = vgg19(pretrained=True)
        self.resnet = resnet50(pretrained=True)

        for params in self.resnet.parameters():
            params.requires_grad = trainable
        
        modules=list(self.resnet.children())[:-2]
        self.resnetSeq=nn.Sequential(*modules) # b, 2048, 14, 14

        self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(14 * 14),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, out_dim),
                nn.BatchNorm1d(14 * 14),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

    def forward(self, x):
        """Converts an image into a feature vector
        
        Args:
            x (N, Channels, Height, Width): Due to vgg, fixed to (3, 448, 448)
        
        Returns:
            feats (N, 14 * 14, out_dim): Feature representation of image. 14 * 14 is fixed due to using of vgg
        """
        feats = self.resnetSeq(x)
        batch_size = x.shape[0]
        feats = feats.view(batch_size, 14 * 14, -1)
        return self.fc(feats)


if __name__ == '__main__':
    model = ImageEncoder(out_dim=640)
    inp = torch.rand((1, 3, 448, 448))
    out = model(inp)
    # assert tuple(out.shape) == (1, 14 * 14, 250)
    summary(model, input = inp)
    print(out.shape)
    print("this is debugger")
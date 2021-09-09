import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append("..")

from image_encoder import ImageEncoder
from question_encoder import QuestionEncoder
from utils.dataset import MultiImageVQADataset


class MultiImageVQA(nn.Module):
    def __init__(self, feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img, n_images=2):
        super(MultiImageVQA, self).__init__()
        self.feat_dim = feat_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.n_images = n_images

        self.img_enc = ImageEncoder(feat_dim)
        self.ques_enc = QuestionEncoder(vocab_size, embed_size, feat_dim)
        self.att1 = nn.MultiheadAttention(feat_dim, 1, batch_first=True)
        self.att2 = nn.MultiheadAttention(feat_dim, 1, batch_first=True)

        self.pred = nn.Linear(feat_dim, vocab_size)


    def forward(self, img_dict, ques):
        """Implements the forward pass for Multi-Image VQA

        Args:
            img_dict (dict): dict with the following keys.
                images (N, num_image, 3, 448, 448): list containing the images in the form of tensors. Images of shape: 448, 448 as required by multi_image_vqa model
            ques (N, seq_len): tensor representing the question

        Returns:
            (N, num_image), (N, vocab_len): Attention weights corresponding to final image selection, and a vocab-dimentional vector representing the generated answer.
        """ 
        images = img_dict['images']
        images_enc = []
        for image in images:
            image = self.img_enc(image)
            image = torch.clamp(image, min=-1, max=-0.5)
            images_enc.append(image)

        # img1 = self.img_enc(img1)
        # img1 = torch.clamp(img1, min=-1, max=-0.5)
        # img2 = self.img_enc(img2)
        # img2 = torch.clamp(img2, min=-1, max=-0.5)
        ques = torch.unsqueeze(self.ques_enc(ques), dim=1)
        # ques = torch.clamp(ques, min=-1, max=-0.5)
        
        # Try Normalizing
        images_att = []
        for image in images_enc:
            image, weight = self.att1(ques, image, image)
            images_att.append((image, weight))
        
        img = torch.stack([image[0] for image in images_att]).squeeze(2)
        ans, weights3 = self.att2(ques, img, img)

        return weights3.squeeze(1), self.pred(ans)


if __name__ == '__main__':
    vocab_size = 9000 # from bert
    seq_len = 12 # from dataset
    feat_dim = 640 # from paper, the final vector vq, vi
    embed_size = 500 # from paper, dimention of embedding of each word
    n_attention_stacks = 2
    hidden_dim_img = feat_dim
    batch_size = 10

    # img_dict = {
    #     'img1': torch.randint(255, size=(2, 3, 448, 448)) / 255,
    #     'img2': torch.randint(255, size=(2, 3, 448, 448)) / 255
    # }

    ques = torch.randint(1000, size=(2, 5))
    model = MultiImageVQA(feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img)

    # out = model(img_dict, ques)
    ds = MultiImageVQADataset()
    dl = DataLoader(ds, batch_size=2)
    # dct = ds[0]
    for dct in dl:
        # for key, val in dct.items():
            # print(f'{key}: {val.shape}', end = ' ')
        # print()
        index, out = model(dct, dct['ques'])
        # print(index)
        # sys.exit(0)
    
    print(out.shape)

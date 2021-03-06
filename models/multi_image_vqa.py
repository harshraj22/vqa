import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append("..")

from image_encoder import ImageEncoder
from question_encoder import QuestionEncoder
from utils.dataset import MultiImageVQADataset

from torchinfo import summary


class MultiImageVQA(nn.Module):
    def __init__(self, feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img):
        super(MultiImageVQA, self).__init__()
        self.feat_dim = feat_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.img_enc = ImageEncoder(feat_dim, trainable=False)
        self.ques_enc = QuestionEncoder(vocab_size, embed_size, feat_dim)
        self.att1 = nn.MultiheadAttention(feat_dim, 1, batch_first=True)
        self.att2 = nn.MultiheadAttention(feat_dim, 1, batch_first=True)

        self.rcnn_feats = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, feat_dim),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(0.1)        
        )

        self.pred = nn.Sequential(
            nn.Linear(feat_dim, 300), # 640 -> 300
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 29)
        )
        # self.pred = nn.Sequential(
        #     nn.Linear(feat_dim, 1000),
        #     nn.Dropout(0.2),
        #     nn.Linear(1000, 2000),
        #     nn.Dropout(0.2),
        #     nn.Linear(2000, 3410)
        # )


    def forward(self, img_dict, ques):
        """Implements the forward pass for Multi-Image VQA

        Args:
            img_dict (dict): dict with the following keys.
                images (N, num_image, 3, 448, 448): list containing the images in the form of tensors. Images of shape: 448, 448 as required by multi_image_vqa model
            ques (N, seq_len, embed_size): tensor representing the question features

        Returns:
            (N, num_image), (N, vocab_len): Attention weights corresponding to final image selection, and a vocab-dimentional vector representing the generated answer.
        """ 
        images = img_dict['images']
        N, num_images, c, h, w = images.shape

        # --------------------------------- Part 0 ---------------------------------
        # image_rcnn_feats.shape: N, num_images, 30, feat_dim
        image_rcnn_feats = self.rcnn_feats(img_dict['image_features_list'])
        assert tuple(image_rcnn_feats.shape) == (N, num_images, 30, self.feat_dim), f"Shapes of image features do not match, expected {(N, num_images, 30, self.feat_dim)}, got {image_rcnn_feats.shape}"

        # --------------------------------- Part 1 ----------------------------------

        # images_enc = []
        # todo: change loop to N * num_image
        images = images.view(N * num_images, c, h, w)
        images = self.img_enc(images)
        # images = torch.clamp(images, min=-1, max=-0.5)
        images_enc = images.view(N, num_images, 196, self.feat_dim)
        # for batch in images:
        #     batch = self.img_enc(batch)
        #     batch = torch.clamp(batch, min=-1, max=-0.5)
        #     images_enc.append(batch)
        # images_enc.shape: (N, num_images, 196, feat_dim)
        questions = torch.unsqueeze(self.ques_enc(ques), dim=1)
        # questions.shape: (N, 1, feat_dim)

        images_enc = torch.cat([images_enc, image_rcnn_feats], dim=2)
        # 226 = 296 + 30
        assert tuple(images_enc.shape) == (N, num_images, 226, self.feat_dim), f"Shapes do not match after the feature extraction of images. Expected: {(N, num_images, 226, feat_dim)}, got: {images.shape}"

        
        # Try Normalizing
        # --------------------------------- Part 2 ----------------------------------

        # for each batch, it stores num_images number of vectors representing the attention output of
        # question, and the corresponding image from the batch
        # len(images_att): batch_size, images_att[0].shape: (num_images, 1, feat_dim)
        # print(f'Images_enc: len: {len(images_enc)}, shape: {images_enc[0].shape}')

        # images_att = []
        # for batch, ques in zip(images_enc, questions):
        #     # for the current batch, find attention with each image and question vector,
        #     # store this result. cur_batch[0].shape: (1, 1, num_features) & len(cur_batch): num_images
        #     cur_batch, ques = [], ques.unsqueeze(0)
        #     # ques.shape: (1, 1, feat_dim)
        #     for image in batch:
        #         image = image.unsqueeze(0)
        #         # image.shape: (1, 196, feat_dim)
        #         image, weights = self.att1(ques, image, image)
        #         cur_batch.append(image)
                
        #     images_att.append(torch.cat(cur_batch, dim=0))
        images_att = []
        # print(f'batch_size {N} Shape of images_enc: {images_enc.shape}')
        images_enc = images_enc.permute(1, 0, 2, 3)
        # print(f'Shape of images_enc after permute: {images_enc.shape}')
        for batch_image in images_enc:
            # loop num_images times
            out, weights = self.att1(questions, batch_image, batch_image)
            images_att.append(out)

        images_att = torch.stack(images_att, dim=0)
        # print(f'Shape of images_att: {images_att.shape}')
        images_att = images_att.permute(1, 0, 2, 3)

        assert len(images_att) == N and tuple(images_att[0].shape) == (num_images, 1, self.feat_dim), f"shapes do not match after the first attention layer. Expected {(N, num_images, 1, self.feat_dim)}, got: {(len(images_att), images_att[0].shape)}"


        # --------------------------------- Part 3 ----------------------------------
        # images_att = torch.cat(images_att, dim=0)
        # images_att.shape: (N, num_images, 1, feat_dim)
        # questions.shape: (N, 1, 1, feat_dim)
        images_att = images_att.squeeze(dim=2)
        questions = questions.squeeze(dim=2)

        # batch_features, batch_weights = [], []
        # for batch, ques in zip(images_att, questions):
        #     image = batch 
        #     ques = ques.unsqueeze(0)
            
        #     ans, weights = self.att2(ques, image, image)
        #     batch_features.append(ans)
        #     batch_weights.append(weights)
        batch_features, batch_weights = self.att2(questions, images_att, images_att)

        # img = torch.stack([image[0] for image in images_att]).squeeze(2)
        # print(f'Pre: , org cat: {torch.cat([image[0] for image in images_att], dim=-2).shape} & Images_att: {images_att[0][0].shape} & {len(images_att)}, images_enc: {images_enc[0].shape}, ques: {ques.shape}')
        # print(f'Before final: ques: {ques.shape}, img: {img.shape}')

        # batch_weights = torch.cat(batch_weights, dim=0).squeeze(1)
        batch_weights = batch_weights.squeeze(1)
        assert tuple(batch_weights.shape) == (N, num_images), "Batch weights shape do not match, "
        # batch_weights.shape: (N, num_images)
        # batch_features = torch.cat(batch_features, dim=0).squeeze(1)
        batch_features = batch_features.squeeze(1)
        assert tuple(batch_features.shape) == (N, self.feat_dim), "Batch features shape do not match"
        # batch_features.shape: (N, feat_dim)

        return batch_weights, self.pred(batch_features)


if __name__ == '__main__':
    num_epochs = 30
    vocab_size = 30000 # from bert
    seq_len = 12 # from dataset
    feat_dim = 640 # from paper, the final vector vq, vi
    embed_size = 768 # from paper, dimention of embedding of each word
    n_attention_stacks = 2
    hidden_dim_img = feat_dim
    batch_size = 32
    lambda_ = 100

    # img_dict = {
    #     'img1': torch.randint(255, size=(2, 3, 448, 448)) / 255,
    #     'img2': torch.randint(255, size=(2, 3, 448, 448)) / 255
    # }

    # ques = torch.randint(1000, size=(2, 5))
    model = MultiImageVQA(feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img)

    # out = model(img_dict, ques)
    ds = MultiImageVQADataset('/home/prabhu/CLEVR_v1.0/questions/questions/CLEVR_val_questions.json', '/home/prabhu/CLEVR_v1.0/images/val', '/home/prabhu/Tiny/tiny-imagenet-200/test/images')
    dl = DataLoader(ds, batch_size=2)
    # dct = ds[0]
    for dct in dl:
        summary(model, input_data=(dct, dct['ques_features']))
        # model(dct, dct['ques'])
        break
    # for dct in dl:
    #     # for key, val in dct.items():
    #         # print(f'{key}: {val.shape}', end = ' ')
    #     # print()
    #     index, out = model(dct, dct['ques'])
    #     # print(index)
    #     # sys.exit(0)
    
    # print(out.shape)

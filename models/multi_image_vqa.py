import torch
import torch.nn as nn

from image_encoder import ImageEncoder
from question_encoder import QuestionEncoder

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
        img1 = img_dict['img1']
        img2 = img_dict['img2']

        img1 = self.img_enc(img1)
        img2 = self.img_enc(img2)
        ques = self.ques_enc(ques).squeeze(0)
        
        # Try Normalizing
        print(f'Before Attention: {img1.shape}, {img2.shape}, {ques.shape}\n\n')
        img1, weights1 = self.att1(ques, img1, img1)
        img2, weights2 = self.att1(ques, img2, img2)
        
        img = torch.stack([img1, img2])
        ans = self.att2(ques, img, img)
        return self.pred(ans)


if __name__ == '__main__':
    vocab_size = 9000 # from bert
    seq_len = 12 # from dataset
    feat_dim = 640 # from paper, the final vector vq, vi
    embed_size = 500 # from paper, dimention of embedding of each word
    n_attention_stacks = 2
    hidden_dim_img = feat_dim
    batch_size = 10

    img_dict = {
        'img1': torch.randint(255, size=(1, 3, 448, 448)) / 255,
        'img2': torch.randint(255, size=(1, 3, 448, 448)) / 255
    }

    ques = torch.randint(1000, size=(1, 5))
    model = MultiImageVQA(feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img)

    out = model(img_dict, ques)
    print(out.shape)

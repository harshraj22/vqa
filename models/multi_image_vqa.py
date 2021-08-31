import torch
import torch.nn as nn

from image_encoder import ImageEncoder
from question_encoder import QuestionEncoder

class MultiImageVQA(nn.Module):
    def __init__(self, feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img):
        super(MultiImageVQA, self).__init__()
        self.feat_dim = feat_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.img_enc = ImageEncoder(feat_dim)
        self.ques_enc = QuestionEncoder(vocab_size, embed_size, feat_dim)
        self.att = nn.ModuleList([
                AttentionLayer(hidden_dim_img, feat_dim, feat_dim) for _ in range(n_attention_stacks)
        ])

        self.pred = nn.Linear(feat_dim, vocab_size)


    def forward(self, img, ques):
        """Returns one word output corresponding to input image and question
        
        Args:
            img (N, 3, 448, 448): Image of given shape, constrained due to use of vgg
            ques (N, seq_len): Question encoded as integers
        
        Returns:
            u (N, vocab_size): Vectors of vocab_size, for picking the most probable word
        """

        img = self.img_enc(img)
        ques = self.ques_enc(ques)
        
        # Try Normalizing
        u = ques
        for layer in self.att:
        u = layer(img, u)
        return self.pred(u)
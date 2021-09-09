from torch.utils.data import Dataset
import json
import torchvision.transforms as T
from PIL import Image 
from transformers import BertTokenizer, DistilBertTokenizer, BertTokenizerFast
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from models.multi_image_vqa import MultiImageVQA
from utils.dataset import MultiImageVQADataset

num_epochs = 30
vocab_size = 30000 # from bert
seq_len = 12 # from dataset
feat_dim = 640 # from paper, the final vector vq, vi
embed_size = 500 # from paper, dimention of embedding of each word
n_attention_stacks = 2
hidden_dim_img = feat_dim
batch_size = 32
lambda_ = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MultiImageVQA(feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img)
model.load_state_dict(torch.load('/nfs_home/janhavi2021/vqa/models/weights.pth'))
for param in model.parameters():
    param.requires_grad = False

model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
ques = tokenizer.tokenize('What is the object ?')
ques = tokenizer.convert_tokens_to_ids(ques)

img_path = '/nfs_home/janhavi2021/textvqa/21069789141_876c38328b_o.jpg'
img = np.array(Image.open(img_path).crop((0, 0, 448, 448)))
img = np.moveaxis(img, -1, 0) / 255
img = torch.from_numpy(img).float()

dct = {
    'images': torch.stack([img, img, img, img]).unsqueeze(0),
    'ques': torch.Tensor(ques).long().unsqueeze(0),
    'ans': torch.Tensor([0]).long(),
    'true_img_index': torch.Tensor([1]).long()
}

for key, val in dct.items():
    print(key, type(val))
print('Ques: ', dct['ques'], type(dct['ques'][0]), type(dct['ques'][0][0]), dct['ques'].shape)


print('\n\n\n')
# out.shape: (N, vocab_size)
attention_weights, out = model(dct, ques)

# assume N = 1
index = torch.argmax(out.squeeze(0))
word = tokenizer.convert_ids_to_tokens([index])
print(f'Generated word: {word}')
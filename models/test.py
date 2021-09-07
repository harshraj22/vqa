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
vocab_size = 9000 # from bert
seq_len = 12 # from dataset
feat_dim = 640 # from paper, the final vector vq, vi
embed_size = 500 # from paper, dimention of embedding of each word
n_attention_stacks = 2
hidden_dim_img = feat_dim
batch_size = 10
lambda_ = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MultiImageVQA(feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
ques = tokenizer.tokenize('What is the object ?')
ques = tokenizer.convert_tokens_to_ids(ques)

img_path = '/nfs_home/janhavi2021/textvqa/21069789141_876c38328b_o.jpg'
img = np.array(Image.open(img_path))
img = np.moveaxis(img, -1, 0)
img = torch.from_numpy(img)

# img

# out.shape: (N, vocab_size)
attention_weights, out = model(img, ques)

# assume N = 1
index = torch.argmax(out.squeeze(0))
word = tokenizer.convert_ids_to_tokens([index])
print(f'Generated word: {word}')
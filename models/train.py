import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append("..")

from models.multi_image_vqa import MultiImageVQA
from utils.dataset import MultiImageVQADataset

num_epochs = 5
vocab_size = 9000 # from bert
seq_len = 12 # from dataset
feat_dim = 640 # from paper, the final vector vq, vi
embed_size = 500 # from paper, dimention of embedding of each word
n_attention_stacks = 2
hidden_dim_img = feat_dim
batch_size = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MultiImageVQA(feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img)
criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

ds = MultiImageVQADataset()
dl = DataLoader(ds, batch_size=5)

# batch = next(dl)
for x in dl:
    batch = x

epoch = 0
for epoch in tqdm(range(num_epochs), desc=f"on epoch {epoch}"):
    out = model(batch, batch['ques'])
    optimizer.zero_grad()
    # pred = torch.argmax(out.squeeze(1), dim=1)
    # print('out: ', out.squeeze(1).shape, 'ans: ', batch['ans'].squeeze(-1).shape, batch['ans'].squeeze(-1).detach().cpu().tolist())
    loss = criterian(out.squeeze(1), batch['ans'].squeeze(-1))
    tqdm.write(f'Loss: {loss.item():.3f}')
    loss.backward()
    optimizer.step()

    # print(f' Ans: {batch["ans"].detach().cpu().tolist()}, ques: {batch["ques"].shape}, out: {out.shape}, pred: {pred.detach().cpu().tolist()}')

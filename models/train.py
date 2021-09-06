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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MultiImageVQA(feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img)
criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=6)

ds = MultiImageVQADataset()
dl = DataLoader(ds, batch_size=5)

# batch = next(dl)
for x in dl:
    batch = x

epoch, losses = 0, []
for epoch in tqdm(range(num_epochs), desc=f"on epoch {epoch}"):
    optimizer.zero_grad()
    attention_weights, out = model(batch, batch['ques'])
    # pred = torch.argmax(out.squeeze(1), dim=1)
    # print('out: ', out.squeeze(1).shape, 'ans: ', batch['ans'].squeeze(-1).shape, batch['ans'].squeeze(-1).detach().cpu().tolist())
    word_loss = criterian(out.squeeze(1), batch['ans'].squeeze(-1))
    img_classification_loss = criterian(attention_weights, batch['true_img_index'])
    tqdm.write(f'word_Loss: {word_loss.item():.3f}, img_classification_loss: {img_classification_loss.item():.3f}, ans_index: {batch["true_img_index"]}')
    loss = word_loss + img_classification_loss
    losses.append(loss.detach().cpu().item())
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    # print(f' Ans: {batch["ans"].detach().cpu().tolist()}, ques: {batch["ques"].shape}, out: {out.shape}, pred: {pred.detach().cpu().tolist()}')

# losses = [3, 4, 2, 5]
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('LossGraph.png')
from PIL.Image import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from math import inf

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
criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=6)



class Multi(Dataset):
    def __init__(self):
        self.n_images = 3

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        dct = {
            f'images': [torch.randint(255, size=(3, 448, 448)) / 255 for x in range(1, self.n_images+1)]
        }

        batch_size, seq_len = 1, 5
        dct['ques'] = torch.tensor([500 for _ in range(seq_len)])  # torch.randint(1000, size=(seq_len,))
        dct['ans'] = torch.tensor([30]) # torch.randint(9000-2, size=(1,))
        dct['true_img_index'] = 1 # random.randint(0, self.n_images-1)
        return dct

ds = MultiImageVQADataset('/nfs_home/janhavi2021/clever/CLEVR_v1.0/questions/CLEVR_val_questions.json', '/nfs_home/janhavi2021/clever/CLEVR_v1.0/images/val', '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/test/images')
datasetLength = len(ds)

print("Dataset Loaded. ", datasetLength)
train_, validation_ = torch.utils.data.random_split(ds, [int(0.9*datasetLength), len(ds) - int(0.9*datasetLength)], generator=torch.Generator().manual_seed(42))
dl_train = DataLoader(train_, batch_size=batch_size, num_workers=4)
dl_val = DataLoader(validation_, batch_size=batch_size, num_workers=4)

# batch = next(dl)
# for inx, x in enumerate(dl):
#     batch = x

# dl_train, dl_val = dl, dl

epoch, train_losses, val_losses, best_val_loss = 0, [], [], inf
for epoch in tqdm(range(num_epochs), desc=f"on epoch {epoch}"):

    for dataloader, phase in [(dl_train, 'train'),(dl_val, 'val')]:
        if phase == 'train':
            model.train()
        elif phase == 'val':
            model.eval()

        phase_loss, phase_word_loss, phase_image_loss = 0, 0, 0
        for batch in tqdm(dataloader, desc="iter"):
            optimizer.zero_grad()
            if phase == 'val':
                with torch.no_grad():
                    attention_weights, out = model(batch, batch['ques'])
            else:
                attention_weights, out = model(batch, batch['ques'])
            # pred = torch.argmax(out.squeeze(1), dim=1)
            # print('out: ', out.squeeze(1).shape, 'ans: ', batch['ans'].squeeze(-1).shape, batch['ans'].squeeze(-1).detach().cpu().tolist())
            word_loss = criterian(out.squeeze(1), batch['ans'].squeeze(-1))
            # print(attention_weights.shape, batch["true_img_index"].shape)
            img_classification_loss = criterian(attention_weights, batch['true_img_index'].squeeze(-1))
            # tqdm.write(f'wetmux ight: {attention_weights}')
            loss = word_loss + lambda_ * img_classification_loss #
            phase_word_loss += word_loss.item() * len(batch['ans'])
            phase_image_loss += img_classification_loss.item() * len(batch['ans'])
            phase_loss += loss.item() * len(batch['ans'])

            if phase == 'train':
                # train_losses.append(loss.detach().cpu().item())
                loss.backward()
                optimizer.step()
            
            elif phase == 'val':
                # val_losses.append()
                scheduler.step(loss)
        
        tqdm.write(f'phase: {phase}, loss: {phase_loss / len(dataloader.dataset):.3f}, word_Loss: {phase_word_loss / len(dataloader.dataset):.3f}, img_classification_loss: {phase_image_loss / len(dataloader.dataset):.3f}')
        if phase == 'train':
            train_losses.append(phase_loss / len(dataloader.dataset))
        else:
            val_losses.append(phase_loss / len(dataloader.dataset))
            if best_val_loss > phase_loss / len(dataloader.dataset):
                best_val_loss = phase_loss / len(dataloader.dataset)
                torch.save(model.state_dict(), 'weights.pth')
            tqdm.write(f'Val Loss: {best_val_loss:.3f}, Saving weights....')

    # print(f' Ans: {batch["ans"].detach().cpu().tolist()}, ques: {batch["ques"].shape}, out: {out.shape}, pred: {pred.detach().cpu().tolist()}')

# losses = [3, 4, 2, 5]
    plt.clf()
    plt.plot(train_losses, color = 'r', label = 'train')
    plt.plot(val_losses, color = 'g', label = 'val')
    plt.title(f'Dataset: {datasetLength}, lambda: {lambda_}')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('LossGraph.png')
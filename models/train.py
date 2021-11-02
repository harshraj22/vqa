from PIL.Image import Image
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import logging
import json
from pprint import pprint
from math import inf
from sklearn.metrics import accuracy_score
from torchinfo import summary
import pdb 
import os
import sys
import time
from datetime import timedelta

sys.path.append("..")

from models.multi_image_vqa import MultiImageVQA
from models.custom_logging import TqdmLoggingHandler
from utils.dataset import MultiImageVQADataset, arrange_batch, VQAOne

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
start_time = time.time()

# ========================================= < Label Smoothing > =======================================
def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss



#  Implementation of Label smoothing with CrossEntropy and ignore_index
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean',ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction,ignore_index=self.ignore_index)
        return linear_combination(loss/n, nll, self.epsilon)

# ========================================= <\ Label Smoothing > =======================================

# ========================================= < Logging > ===========================
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())

# ========================================= </ Logging > ==========================
num_epochs = 180
vocab_size = 30000 # from bert
seq_len = 12 # from dataset
feat_dim = 640 # from paper, the final vector vq, vi
embed_size = 768 # from paper, dimention of embedding of each word
n_attention_stacks = 2
hidden_dim_img = feat_dim
batch_size = 20
num_workers = 20
lambda_ = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

answers = ['6', 'yes', 'small', 'blue', 'brown', 'red', 'cyan', '1', '0', 'rubber', '7', 'no', 'purple', 'large', '5', 'green', '10', 'cube', '4', '9', '3', 'cylinder', 'metal', '2', 'gray', 'sphere', '8', 'yellow']

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

ds = MultiImageVQADataset('/home/prabhu/vqa/models/cleaned.json', '/home/prabhu/CLEVR_v1.0/images/val', '/home/prabhu/Tiny/tiny-imagenet-200/test/images')
# ds = VQAOne("/home/prabhu/VQA1.0/v2_mscoco_train2014_annotations_cleaned.json", "/home/prabhu/VQA1.0/v2_OpenEnded_mscoco_train2014_questions.json", "/home/prabhu/VQA1.0/train2014")
datasetLength = len(ds)

# answer_tokens = ds.tokenizer.tokenize(answers)
# answer_indices = ds.tokenizer.convert_tokens_to_ids(answer_tokens)
# crossentropy_weights = [0 for _ in range(30000)]
# for answer_index in answer_indices:
#     crossentropy_weights[answer_index] = 1 / len(answer_indices)

print("Dataset Loaded. ", datasetLength)
# pdb.set_trace()
train_, validation_ = torch.utils.data.random_split(ds, [int(0.8*datasetLength), len(ds) - int(0.8*datasetLength)], generator=torch.Generator().manual_seed(42))
dl_train = DataLoader(train_, batch_size=batch_size, num_workers=num_workers, collate_fn=arrange_batch, shuffle=True)
dl_val = DataLoader(validation_, batch_size=batch_size+10, num_workers=num_workers, collate_fn=arrange_batch, shuffle=True)


model = MultiImageVQA(feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img)
state_dict = torch.load('/home/prabhu/vqa/models/exp89/models/exp0/weights.pth')
# # print(*state_dict.keys(), sep='\n')
# del state_dict['pred.4.weight']
# del state_dict['pred.4.bias']
model.load_state_dict(state_dict['state_dict'], strict=False)
print(f'Loaded model weights....')
model = model.to(device)
# model = nn.DataParallel(model)

criterian = nn.CrossEntropyLoss() # LabelSmoothingCrossEntropy() #
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=6, min_lr=1e-6, factor=0.5)


# ============================== < Set up the project dir > =============================

PROJECT_DIR = '/home/prabhu/vqa/models'
counts = list(filter(lambda x: x.startswith("exp"), os.listdir(PROJECT_DIR)))
# print(counts)
counts = list(map(lambda x: int(x.replace('exp', '')), counts))

EXP_DIR = f'exp{max(counts) + 1}'

os.mkdir(os.path.join(PROJECT_DIR, EXP_DIR))
# ============================== <\ Set up the project dir > =============================

# batch = next(dl)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, end=' ')

# for inx, x in enumerate(dl_train):
#     batch = x
#     break
# summary(model, input_data=(batch, batch['ques_features']))
# exit(0)

# dl_train, dl_val = dl, dl

IMG_CLS_CONST, IMG_CLS_FACTOR = 600, 0.5

epoch, train_losses, val_losses, best_val_loss = 0, [], [], inf
word_accuracy, image_accuracy = {'train': [], 'val': []}, {'train': [], 'val': []}
losses = {
    'train': {
        'image_loss': [],
        'word_loss': []
    },
    'val': {
        'image_loss': [],
        'word_loss': []
    }
}
for epoch in tqdm(range(num_epochs), desc=f"on epoch {epoch}"):

    for dataloader, phase in [(dl_train, 'train'),(dl_val, 'val')]:
        if phase == 'train':
            model.train()
        elif phase == 'val':
            model.eval()

        phase_loss, phase_word_loss, phase_image_loss = 0, 0, 0
        phase_word_accuracy, phase_image_accuracy = 0, 0
        for batch in tqdm(dataloader, desc=f"iter {phase} | time: {timedelta(seconds=time.time() - start_time)}"):
            # batch = x
            optimizer.zero_grad()
            keys = batch.keys()

            for key in keys:
                batch[key] = batch[key].to(device)

            if phase == 'val':
                with torch.no_grad():
                    attention_weights, out = model(batch, batch['ques_features'])
            else:
                attention_weights, out = model(batch, batch['ques_features'])

            # log.info(attention_weights)
            pred = torch.argmax(out, dim=1)
            tqdm.write(f'pred: {pred.clone().detach().cpu().tolist()} {torch.argmax(attention_weights, dim=1).clone().detach().cpu().tolist()}\nAns:  {batch["ans"].squeeze(-1).clone().detach().cpu().tolist()} {batch["true_img_index"].squeeze(-1).clone().detach().cpu().tolist()}')
            # tqdm.write(f'Debug: {attention_weights.shape}, {batch["true_img_index"]}')
            # print('out: ', out.shape, 'ans: ', batch['ans'].shape, batch['ans'].detach().cpu().tolist())
            word_loss = criterian(out, batch['ans'].squeeze(-1))
            # torch.tensor([0]).to(device) # 
            img_classification_loss = criterian(attention_weights, batch['true_img_index'].squeeze(-1))
            #tqdm.write(f'wetmux ight: {attention_weights}')
            #continue
            loss = (lambda_ * word_loss) # + (IMG_CLS_CONST * img_classification_loss)
            phase_word_loss += lambda_ * word_loss.item() * len(batch['ans'])
            phase_image_loss += img_classification_loss.item() * len(batch['ans'])
            phase_loss += loss.item() * len(batch['ans'])

            phase_word_accuracy += accuracy_score(
                batch['ans'].squeeze(-1).clone().detach().cpu().tolist(),
                pred.clone().detach().cpu().tolist(),
                normalize=False
                )

            phase_image_accuracy += accuracy_score(
                batch["true_img_index"].squeeze(-1).clone().detach().cpu().tolist(),
                torch.argmax(attention_weights, dim=1).clone().detach().cpu().tolist(),
                normalize=False
                )

            if phase == 'train':
                # train_losses.append(loss.detach().cpu().item())
                loss.backward()
                optimizer.step()
            
            elif phase == 'val':
                pass
                # val_losses.append()
                # scheduler.step(loss)

        tqdm.write(f'phase: {phase} | loss: {phase_loss / len(dataloader.dataset):.3f}, word_Loss: {phase_word_loss / len(dataloader.dataset):.3f}, img_cls_loss: {phase_image_loss / len(dataloader.dataset):.3f}, img_cls_cnst: {IMG_CLS_CONST} | word_accuracy: {phase_word_accuracy / len(dataloader.dataset):.3f} | LR: {optimizer.param_groups[0]["lr"]}')
        
        word_accuracy[phase].append(phase_word_accuracy / len(dataloader.dataset))
        image_accuracy[phase].append(phase_image_accuracy / len(dataloader.dataset))

        losses[phase]['word_loss'].append(phase_word_loss / len(dataloader.dataset))
        losses[phase]['image_loss'].append(phase_image_loss / len(dataloader.dataset))

        if phase == 'train':
            train_losses.append(phase_loss / len(dataloader.dataset))
        else:
            val_losses.append(phase_loss / len(dataloader.dataset))
            if best_val_loss > phase_loss / len(dataloader.dataset):
                best_val_loss = phase_loss / len(dataloader.dataset)
                torch.save({'state_dict': model.state_dict(), 'lr': optimizer.param_groups[0]["lr"]}, os.path.join(EXP_DIR, 'weights.pth'))
            scheduler.step(phase_loss)
            tqdm.write(f'Val Loss: {best_val_loss:.3f}, Saving weights....')

    IMG_CLS_CONST = IMG_CLS_CONST * IMG_CLS_FACTOR
    IMG_CLS_CONST = max(IMG_CLS_CONST, 1.0)
    # print(f' Ans: {batch["ans"].detach().cpu().tolist()}, ques: {batch["ques"].shape}, out: {out.shape}, pred: {pred.detach().cpu().tolist()}')

# losses = [3, 4, 2, 5]
    plt.clf()
    plt.plot(train_losses, color = 'r', label = 'train')
    plt.plot(val_losses, color = 'g', label = 'val')
    plt.title(f'Dataset: {datasetLength}, lambda: {lambda_}')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(EXP_DIR, 'LossGraph.png'))


    with open(os.path.join(EXP_DIR, 'losses.json'), 'w') as f:
        json.dump(
            {'word_accuracy': word_accuracy,
            'image_accuracy': image_accuracy,
            'losses': losses
            },
            f,
            indent=4
        )
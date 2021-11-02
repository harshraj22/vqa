# Caution ! Not updated !!!
from torch.utils.data import Dataset
import json
import torchvision.transforms as T
from PIL import Image 
import numpy as np
from transformers import BertTokenizer, DistilBertTokenizer, BertTokenizerFast, BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchinfo import summary

import sys
sys.path.append("..")

from models.multi_image_vqa import MultiImageVQA
from utils.dataset import MultiImageVQADataset

import warnings
warnings.filterwarnings('ignore')

def get_features(ques_str, tokenizer, max_seq_len):
    bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=False)
    bert.eval()
    # returns features corresponding to the question string
    tokens = tokenizer.tokenize(ques_str)
    
    # questokenized = self.tokenizer.tokenize(ques['question'])
    prefix = ['[PAD]' for i in range(max_seq_len - min(max_seq_len, len(tokens)))]
    prefix = ' '.join(prefix)

    # dct['ques_features'] = self._get_features(ques['question'] + prefix)
    questokenized = tokenizer.tokenize(ques_str + prefix)[:max_seq_len] 

    indexes = tokenizer.convert_tokens_to_ids(questokenized)

    # indexes = self.tokenizer.convert_tokens_to_ids(tokens)
    index_tensors = torch.tensor(indexes).unsqueeze(0)
    segment_idx = torch.zeros(index_tensors.shape)

    outs = bert(index_tensors, segment_idx)[0][0]
    # outs = torch.tensor(outs)
    # print(len(outs), len())
    assert tuple(outs.shape) == (len(indexes), 768), "Error while extracting bert features."
    return outs.detach()

def inference(pathList, ques, max_seq_len=30):
    dct = {}
    answers = ['6', 'yes', 'small', 'blue', 'brown', 'red', 'cyan', '1', '0', 'rubber', '7', 'no', 'purple', 'large', '5', 'green', '10', 'cube', '4', '9', '3', 'cylinder', 'metal', '2', 'gray', 'sphere', '8', 'yellow']

    preprocess = T.Compose([
                        T.Resize(512),
                        T.CenterCrop(448),
                        T.ToTensor(),
                        T.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                )
                        ])
    imageList = []
    featList = []

    for path in pathList:
        img = Image.open(path).convert('RGB') 
        img = preprocess(img)
        imageList.append(img)

        # featList.append(torch.load(path.split('.')[0] + '.pth'))
        featList.append(torch.load(path.replace('png', 'pth').replace('JPEG', 'pth')))

    dct['images'] = torch.stack(imageList, dim=0)
    dct['image_features_list'] = torch.stack(featList, dim=0)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    questokenized = tokenizer.tokenize(ques)
    prefix = ['[PAD]' for i in range(max_seq_len - min(max_seq_len, len(questokenized)))]
    prefix = ' '.join(prefix)

    with torch.no_grad():
        dct['ques_features'] = get_features(ques + prefix, tokenizer, max_seq_len)

    for key, value in dct.items():
        dct[key] = value.unsqueeze(0)
    
    feat_dim = 640
    vocab_size = 30000
    embed_size = 768
    n_attention_stacks = 2
    hidden_dim_img = feat_dim

    model = MultiImageVQA(feat_dim, vocab_size, embed_size, n_attention_stacks, hidden_dim_img)
    state_dict = torch.load('/home/prabhu/vqa/models/exp89/models/exp0/weights.pth')
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    model.cpu()
    for key, val in dct.items():
        dct[key] = val.cpu()

    with torch.no_grad():
        att, output = model(dct, dct['ques_features'])
    pred = torch.argmax(output, dim=1)
    print(f'Word index pred: {pred.clone().detach().cpu().tolist()}, Att Weights: {torch.argmax(att, dim=1).clone().detach().cpu().tolist()}\n')

    print("Answer: ", answers[pred[0].item()])
 
    
if __name__ == '__main__':
    inference(["/home/prabhu/Tiny/tiny-imagenet-200/test/images/test_1.JPEG", "/home/prabhu/Tiny/tiny-imagenet-200/test/images/test_0.JPEG", "/home/prabhu/CLEVR_v1.0/images/val/CLEVR_val_000003.png", "/home/prabhu/Tiny/tiny-imagenet-200/test/images/test_3.JPEG"], "How many yellow balls are there?")

if __name__ == "123__main__":
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
    # for param in model.parameters():
    #     param.requires_grad = False

    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ques = tokenizer.tokenize('wghjk ghjk kjhg kjhg ?')
    ques = tokenizer.convert_tokens_to_ids(ques)

    def get_image(img_path = '/nfs_home/janhavi2021/clever/CLEVR_v1.0/images/val/CLEVR_val_000000.png'):
        #'/nfs_home/janhavi2021/textvqa/20605099204_4fe191a2e8_o (1).jpg'
        # for images from CLEVER dataset, we need to convert
        if 'CLEVR' in img_path:
            img = np.array(Image.open(img_path).crop((0, 0, 448, 448)).convert('RGB'))
        else:
            img = np.array(Image.open(img_path).crop((0, 0, 448, 448)))
        img = np.moveaxis(img, -1, 0) / 255
        img = torch.from_numpy(img).float()
        return img

    # img_paths = ['/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/val/images/val_5.JPEG', '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/val/images/val_19.JPEG', '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/val/images/val_38.JPEG', '/nfs_home/janhavi2021/clever/CLEVR_v1.0/images/val/CLEVR_val_000000.png']

    img_paths = [
        '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/val/images/val_30.JPEG', '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/val/images/val_142.JPEG', '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/val/images/val_1.JPEG', '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/val/images/val_5.JPEG'
    ][::-1]

    print(f'Inside test: {torch.stack([get_image(x) for x in img_paths]).unsqueeze(0).shape}')

    dct = {
        'images': torch.stack([get_image(x) for x in img_paths]).unsqueeze(0),
        'ques': torch.Tensor(ques).long().unsqueeze(0),
        'ans': torch.Tensor([0]).long(),
        'true_img_index': torch.Tensor([1]).long()
    }

    for key, val in dct.items():
        print(key, type(val))
    # print('Ques: ', dct['ques'], type(dct['ques'][0]), type(dct['ques'][0][0]), dct['ques'].shape)


    print('\n\n\n')
    # out.shape: (N, vocab_size)
    model.train()
    summary(model, input_data = (dct,dct['ques']))
    model.eval()
    attention_weights, out = model(dct, dct['ques'])
    print(f'Max weights belong to: {torch.argmax(attention_weights)}, {attention_weights}, Out: {out.shape}')

    # assume N = 1
    # print(out.squeeze(0), out.squeeze(0).shape)
    index = torch.argmax(out.squeeze(0))
    word = tokenizer.convert_ids_to_tokens([index])
    print(f'Generated word: {word}')
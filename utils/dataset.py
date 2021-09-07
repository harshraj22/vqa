import torch 
import torch.nn
import random
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import torchvision.transforms as T
import numpy as np
import json
from transformers import BertTokenizer, DistilBertTokenizer, BertTokenizerFast

class MultiImageVQADataset(Dataset):
    """
    one image from clever and other 3 image from other.

    """
    def __init__(self, cleverTrainJson, cleverPath, tinyPath, randomImages=3, max_seq_len=30):
        # generating #randomImages extra images from Clever for each inx
        self.cleverPath = cleverPath
        self.max_seq_len = max_seq_len

        with open(cleverTrainJson) as f:
            data = json.load(f)

        self.questions = data["questions"] # 70l
        self.length = len(self.questions)

        self.tinyPath = tinyPath 
        self.tinyImages = os.listdir(self.tinyPath) 

        self.randomImages = randomImages
        self.randomIndices = {}

        for i in range(self.length):
            self.randomIndices.update({i: random.sample(self.tinyImages, self.randomImages)})

        self.preprocess = T.Compose([
                                    T.Resize(512),
                                    T.CenterCrop(448),
                                    T.ToTensor(),
                                    T.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                    )
                        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return self.length // 1000

    def __getitem__(self, idx):

        ques = self.questions[idx]

        originalImage = Image.open(os.path.join(self.cleverPath, ques['image_filename'])).convert('RGB') #torch.randint(255, size=(3, 448, 448)) / 255
        originalImage = self.preprocess(originalImage)

        assert tuple(originalImage.shape) == (3, 448, 448), f"Image size issue {originalImage.shape}."
        trueImageIndex = random.sample([i for i in range(self.randomImages+1)], 1)[0]

        imageList = []

        for image in self.randomIndices[idx]:
            img = Image.open(os.path.join(self.tinyPath, image)).convert('RGB')
            img = self.preprocess(img)
            imageList.append(img)

        imageList.insert(trueImageIndex, originalImage)

        batch_size, seq_len = 1, 5
        dct = {}
        dct['images'] = imageList

        questokenized = self.tokenizer.tokenize(ques['question'])
        prefix = ['[PAD]' for i in range(self.max_seq_len - min(self.max_seq_len, len(questokenized)))]
        prefix = ' '.join(prefix)
        questokenized = self.tokenizer.tokenize(ques['question'] + prefix)[:self.max_seq_len] 

        tokens = self.tokenizer.convert_tokens_to_ids(questokenized)
        dct['ques'] = torch.Tensor(tokens).long()

        anstokenized = self.tokenizer.tokenize(ques['answer'])
        tokens = self.tokenizer.convert_tokens_to_ids(anstokenized)
        dct['ans'] = torch.Tensor([tokens[0]]).long() #torch.randint(9000-2, size=(1,))

        dct['true_img_index'] = torch.Tensor([trueImageIndex]).long()
        
        return dct

if __name__ == '__main__':
    data = MultiImageVQADataset('/nfs_home/janhavi2021/clever/CLEVR_v1.0/questions/CLEVR_val_questions.json', '/nfs_home/janhavi2021/clever/CLEVR_v1.0/images/val', '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/test/images')
    d = data.__getitem__(2)
    print(d['images'][0].shape, len(d['images']))
    print(d["ques"])
    print(d["ans"])

    print(len(d))

    for i in range(1000):
        # print(len(d['images']), len(d['ques']), len(d['ans']))
        for key, val in d.items():
            print(key, len(val), end=' ')
        print()


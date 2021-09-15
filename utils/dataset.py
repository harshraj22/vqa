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
import unittest

class MultiImageVQADataset(Dataset):
    """one image from clever and other 3 image from other.
    """
    def __init__(self, cleverTrainJson, cleverPath, tinyPath, randomImages=3, max_seq_len=30):
        # generating #randomImages extra images from Clever for each inx
        self.cleverPath = cleverPath
        self.max_seq_len = max_seq_len

        with open(cleverTrainJson) as f:
            data = json.load(f)

        self.questions = data["questions"] # 70l
        random.shuffle(self.questions)
        self.length = len(self.questions)

        self.possible_answers = ['6', 'yes', 'small', 'blue', 'brown', 'red', 'cyan', '1', '0', 'rubber', '7', 'no', 'purple', 'large', '5', 'green', '10', 'cube', '4', '9', '3', 'cylinder', 'metal', '2', 'gray', 'sphere', '8', 'yellow']

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
        return self.length // 50

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

        _, seq_len = 1, 5
        dct = {}
        dct['images'] = torch.stack(imageList, dim=0)

        questokenized = self.tokenizer.tokenize(ques['question'])
        prefix = ['[PAD]' for i in range(self.max_seq_len - min(self.max_seq_len, len(questokenized)))]
        prefix = ' '.join(prefix)
        questokenized = self.tokenizer.tokenize(ques['question'] + prefix)[:self.max_seq_len] 

        tokens = self.tokenizer.convert_tokens_to_ids(questokenized)
        dct['ques'] = torch.Tensor(tokens).long()

        # anstokenized = self.tokenizer.tokenize(ques['answer'])
        # tokens = self.tokenizer.convert_tokens_to_ids(anstokenized)
        # dct['ans'] = torch.Tensor([tokens[0]]).long() #torch.randint(9000-2, size=(1,))
        if ques['answer'] in self.possible_answers:
            dct['ans'] = torch.Tensor([self.possible_answers.index(ques['answer']) ]).long()
        else:
            dct['ans'] = torch.Tensor([len(self.possible_answers)])

        dct['true_img_index'] = torch.Tensor([trueImageIndex]).long()
        
        return dct


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_random_images = 3
        print(f'Creating the dataset......')
        self.data = MultiImageVQADataset('/nfs_home/janhavi2021/clever/CLEVR_v1.0/questions/CLEVR_val_questions.json', '/nfs_home/janhavi2021/clever/CLEVR_v1.0/images/val', '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/test/images', randomImages=self.num_random_images)
        print(f'Created the dataset....')

    def test_dct_len(self):
        # num of points in dataset should be non-zero
        self.assertTrue(len(self.data) > 0)

    def test_return_type(self):
        # the objects of dataset should be dictionaries
        dct = self.data[0]
        self.assertEqual(type(dct), type(dict()))

    def test_dct_keys(self):
        # some keys must be present in the dictionary.
        # these keys are used in the training loop, and the model
        dct = self.data[0]
        required_keys = ['images', 'ans', 'ques', 'true_img_index']
        self.assertTrue(all(key in dct.keys() for key in required_keys))

    def test_num_images(self):
        dct = self.data[0]
        self.assertEqual(len(dct['images']), self.num_random_images + 1)

    def test_one_word_answer(self):
        # the answer returned, should be of one word
        dct = self.data[0]
        self.assertEqual(tuple(dct['ans'].shape), (1,))

    def test_answer_dtype(self):
        # answer should be torch.LongTensor
        dct = self.data[0]
        self.assertEqual(dct['ans'].dtype, torch.int64)

    def test_ques_dtype(self):
        # ques should be torch.LongTensor
        dct = self.data[0]
        self.assertEqual(dct['ques'].dtype, torch.int64)

    def test_img_index_dtype(self):
        # image index should be torch.LongTensor
        dct = self.data[0]
        self.assertEqual(dct['true_img_index'].dtype, torch.int64)


def arrange_batch(batch):
    keys = batch[0].keys()
    new_batch = {}

    for key in keys:
        new_batch[key] = torch.stack([batch_item[key] for batch_item in batch], dim=0)

    # for key, val in new_batch.items():
    #     if isinstance(val, list):
    #         print(key, len(val), val[0].shape)
    #     else:
    #         print(key, val.shape)
    return new_batch

if __name__ == '__main__':
    # unittest.main()
    data = MultiImageVQADataset('/nfs_home/janhavi2021/vqa/models/cleaned.json', '/nfs_home/janhavi2021/clever/CLEVR_v1.0/images/val', '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/test/images')
    dl = DataLoader(data, batch_size=2, collate_fn=arrange_batch)
    for batch in dl:
        # print(batch)
        break
        # for key, val in batch.items():
        #     print(key, val.shape)

    # d = data.__getitem__(2)
    # print(d['images'][0].shape, len(d['images']))
    # print(d["ques"])
    # print(d["ans"])

    # print(len(d))

    # for i in range(1000):
    #     # print(len(d['images']), len(d['ques']), len(d['ans']))
    #     for key, val in d.items():
    #         print(key, len(val), end=' ')
    #     print()


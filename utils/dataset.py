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
from transformers import BertTokenizer, DistilBertTokenizer, BertTokenizerFast, BertModel
import unittest

import time
from datetime import timedelta
from torch.utils import data
import nltk
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
TODO:
1. Balance the answers in the VQAOne dataset class.
2. Iterating over the dataset seems too slow. check if some optimization can be
    done.
"""



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
        self.tinyImages = list(filter(lambda x: x.endswith('.JPEG'), os.listdir(self.tinyPath))) 

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
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=False)
        self.bert.eval()

    def _get_features(self, ques_str, return_indexes=False):
        # returns features corresponding to the question string
        tokens = self.tokenizer.tokenize(ques_str)
        
        # questokenized = self.tokenizer.tokenize(ques['question'])
        prefix = ['[PAD]' for i in range(self.max_seq_len - min(self.max_seq_len, len(tokens)))]
        prefix = ' '.join(prefix)

        # dct['ques_features'] = self._get_features(ques['question'] + prefix)
        questokenized = self.tokenizer.tokenize(ques_str + prefix)[:self.max_seq_len] 

        indexes = self.tokenizer.convert_tokens_to_ids(questokenized)

        # ====
        # indexes = self.tokenizer.convert_tokens_to_ids(tokens)
        index_tensors = torch.tensor(indexes).unsqueeze(0)
        segment_idx = torch.zeros(index_tensors.shape)

        outs = self.bert(index_tensors, segment_idx)[0][0]
        # outs = torch.tensor(outs)
        # print(len(outs), len())
        assert tuple(outs.shape) == (len(indexes), 768), "Error while extracting bert features."
        if return_indexes:
            return outs.detach(), index_tensors.detach()

        return outs.detach()

    def __len__(self):
        return self.length #// 500

    def __getitem__(self, idx):

        ques = self.questions[idx]

        originalImage = Image.open(os.path.join(self.cleverPath, ques['image_filename'])).convert('RGB') #torch.randint(255, size=(3, 448, 448)) / 255
        originalImage = self.preprocess(originalImage)

        assert tuple(originalImage.shape) == (3, 448, 448), f"Image size issue {originalImage.shape}."
        trueImageIndex = 0 # random.sample([i for i in range(self.randomImages+1)], 1)[0]

        imageList, image_features_list = [], []

        for image in self.randomIndices[idx]:
            img = Image.open(os.path.join(self.tinyPath, image)).convert('RGB')
            img = self.preprocess(img)
            imageList.append(img)
            # print(f"Loading: {os.path.join(self.tinyPath, image).replace('.JPEG', '.pth')}")
            image_features_list.append(torch.load(os.path.join(self.tinyPath, image).replace('.JPEG', '.pth'), map_location=torch.device('cpu')))

        imageList.insert(trueImageIndex, originalImage)
        # imageList = [originalImage, originalImage, originalImage, originalImage]
        # print(f"Loading Final Image: {os.path.join(self.cleverPath, ques['image_filename']).replace('.png', '.pth')}")
        image_features_list.insert(trueImageIndex, torch.load(os.path.join(self.cleverPath, ques['image_filename']).replace('.png', '.pth'), map_location=torch.device('cpu')))

        _, seq_len = 1, 5
        dct = {}
        dct['images'] = torch.stack(imageList, dim=0)
        dct['image_features_list'] = torch.stack(image_features_list, dim=0)

        questokenized = self.tokenizer.tokenize(ques['question'])
        prefix = ['[PAD]' for i in range(self.max_seq_len - min(self.max_seq_len, len(questokenized)))]
        prefix = ' '.join(prefix)

        with torch.no_grad():
            dct['ques_features'], dct['ques_indexes'] = self._get_features(ques['question'] + prefix, return_indexes=True)
        # questokenized = self.tokenizer.tokenize(ques['question'] + prefix)[:self.max_seq_len] 

        # tokens = self.tokenizer.convert_tokens_to_ids(questokenized)
        # dct['ques'] = torch.Tensor(tokens).long()

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
        self.data = MultiImageVQADataset('/home/prabhu/CLEVR_v1.0/questions/CLEVR_val_questions.json', '/home/prabhu/CLEVR_v1.0/images/val', '/home/prabhu/Tiny/tiny-imagenet-200/test/images', randomImages=self.num_random_images)
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


class ImageDataset:
    def __init__(self):
        self.CLASS_FILES = '/home/prabhu/Caltech256/256_ObjectCategories/'
        self.images = self._load_caltech256()

    def _load_caltech256(self):
        files = os.listdir(self.CLASS_FILES)

        classes = dict()
        for file in files:
            _classes = file.split('.')[-1].split('-')
            for _class in _classes:
                if len(_class) > 2:
                    classes[_class] = self.CLASS_FILES + file
        return classes

    def get_images(self, classes, num_images):
        """Get the images from caltech, sampled randomly such that they do not
        belong to the classes passed as argument

        Args:
            classes (List[String]): List of classes to be ignored
            num_images (Int): Number of images to be returned

        Returns:
            List[String]: List containing paths of the images
        """
        classes_allowed = list(self.images.keys() - set(classes))
        images = []
        for _ in range(num_images):
            # randomly choose a class
            cur_class = random.choice(classes_allowed)
            files = [file for file in os.listdir(self.images[cur_class]) if file.endswith('jpg')]
            # randomly choose a image from the class
            file_path = random.choice(files)
            images.append(self.images[cur_class] + '/' + file_path)

        return images


class VQAOne(Dataset):
    def __init__(self, json_annotation_path, json_question_path, json_image_path, max_seq_len=30):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.suffix = "/home/prabhu/VQA1.0/train2014/COCO_train2014_"  # length of integer after this is 12

        self.image_list = os.listdir(json_image_path)
        answer_freq = defaultdict(lambda: 0)
        self.lemma = nltk.wordnet.WordNetLemmatizer()
        self.image_dataset = ImageDataset()

        with open(json_annotation_path, 'r') as annotation_file:
            annotation_data = json.load(annotation_file)

        with open(json_question_path, 'r') as question_file:
            question_data = json.load(question_file)

        # self.annotations.keys(): [ques_id, img_id, ques_type, ans_type, answer, mcq_ans]
        self.annotations = annotation_data['annotations']
        # self.questions.keys(): [ques_id, img_id, question]
        self.questions = question_data['questions']

        # ============<converting questions json to dictionary>================
        self.questions_dict = dict()
        for q in self.questions:
            self.questions_dict.update({q['question_id']: {
                                                        "img": q['image_id'],
                                                        "que": q['question']
                                                        }})
        # ============</converting questions json to dictionary>===============

        # ==========================<creating answer vocab>====================
        self.ans_vocab = dict()  # dictionary of format string : integer
        index_ = 0

        # an annotation is a datapoint, ie. a question and other attributes
        # related to it, like answers, image, other metadata etc
        for annotation in self.annotations:
            # for each question, there are a set of possible answers
            answer_list = annotation["answers"]
            for l in answer_list:
                # the answers are not always cleaned. eg the following is also
                # present as answers:
                #   "white,grey,brown"
                # which clearly corresponds to 3 words instead of 1
                answer = l["answer"] + ' |'
                answer = self._extract_answer(answer)
                if answer not in self.ans_vocab.keys():
                    self.ans_vocab.update({answer: index_})
                    index_ += 1

                answer_freq[answer] += 1
        # =======================</making answer vocab>========================

        # =======================<Removing questions>==========================
        # this loop repeats the code, and thus voilates DRY principle.
        # would have to refactor it.
        self.annotations_ = []
        for annotation in self.annotations:
            answer_list = annotation['answers']
            for answer in answer_list:
                answer = answer['answer']
                if len(answer.split()) == 1:
                    self.annotations_.append(annotation)
                    break

        self.annotations = self.annotations_
        # =======================</Removing questions>=========================

        self.length = len(self.annotations)

        # Remove those words from Answer-Vocabulary whose frequency is less than
        # 0.01% of the total dataset size. This would help us again to shrink
        # the size of vocab
        for word, freq in answer_freq.items():
            if freq < 0.01 * self.length / 100:
                # print(f'Deleting {word} with freq: {freq}')
                del self.ans_vocab[word]

        # plt.plot([answer_freq[word] for word in self.ans_vocab])
        # plt.savefig('Answer_frequencies.png')
        # total_freq = sum(ans)
        for word in sorted(self.ans_vocab.keys(), key=lambda x: answer_freq[x]):
            with open('ans_words.txt', 'a') as f:
                f.write(f'{word}: {answer_freq[word]}\n')
            # print(f'{word}: {answer_freq[word]}')

        for index, word in enumerate(self.ans_vocab.keys()):
            self.ans_vocab[word] = index
        vocab_length, ans_vocab = len(self.ans_vocab), self.ans_vocab
        self.ans_vocab = defaultdict(lambda: vocab_length)
        self.ans_vocab.update(ans_vocab)

        print('Max Frequencies: ', max([answer_freq[word] for word in self.ans_vocab]))
        print("And total size of answer vocab is : ", len(self.ans_vocab), 'And size of dataset: ', self.length)  # 27233

    def _extract_answer(self, answer):
        """extracts the one word root form of answer from a given string with
        answer(s). eg. 
            input: "plays,runs"
            output: "play"

        Args:
            answer (String): answer string taken from Annotation file

        Returns:
            String: Lemmatized answer string
        """
        # answer can be containing multiple words joint by special symbols. eg
        #   "white,brown,bull"
        for ch in "\.?,/-!'":
            answer = answer.replace(ch, ' ')
        # we consider only the first answer in such cases
        answer = answer.split()[0]
        # we lemmatize the words so that similar words map to same
        # index in the vocab, thus reducing the size of vocab
        # eg. 'their' and 'theirs' are mapped to same index
        answer = self.lemma.lemmatize(answer)
        return answer

    def __len__(self):
        return self.length
        # return 25000

    def convert_imgpath_to_array(self, img_path, img_shape=(448, 448)):
        """returns numpy array corresponding to the image with given path

        Args:
            img_path (String): Absolute path of image
            img_shape (tuple, optional): Height and width of image to be
                resized to. Defaults to (600, 600).

        Returns:
            np.array: numpy array corresponding to the image.
        """
        img = Image.open(img_path).resize(img_shape)
        # img = np.array(torch.Tensor(np.array(Image.open(image_path))).transpose(0, 2))
        img = np.array(img)
        # reshape from (h, w, c) to (c, h, w)
        img = np.moveaxis(img, -1, 0)

        if img.shape == img_shape:
            # if image is black & white, make it 3 channeled one by repeating
            # it multiple times.
            img = np.stack([img, img, img])
        # tqdm.write(f'{img_path}, {img.shape}')
        assert img.shape == (3, *img_shape), "Image resizing failed. VQAOne.convert_imgpath_to_array"
        return img / 225

    def __getitem__(self, inx):
        item = self.annotations[inx]
        question = self.questions_dict[item['question_id']]['que']
        image_id = self.questions_dict[item['question_id']]['img']

        # image_id should be 12 chars long. So pad with '0'
        image_id = str(image_id).rjust(12, '0')

        image_path = self.suffix + image_id + ".jpg"
        image = self.convert_imgpath_to_array(image_path)

        other_images = self.image_dataset.get_images(item['classes'], 3)
        images = [image, image, image, image]

        for index, other_image in enumerate(other_images):
            images[index+1] = self.convert_imgpath_to_array(other_image)
 
        true_index = random.randint(0, 3)
        images[0], images[true_index] = images[true_index], images[0]
        dct = {"images": torch.Tensor(images)}
        dct['true_img_index'] = torch.Tensor([true_index]).long()

        # =============<preparing question in required format>=============
        ques_tokenized = self.tokenizer.tokenize(question)
        # Lenght of question should be fixed, so pad the remaining space with
        # special tokens
        prefix = ['[PAD]' for i in range(self.max_seq_len - min(self.max_seq_len, len(ques_tokenized)))]
        prefix = ' '.join(prefix)
        ques_tokenized = self.tokenizer.tokenize(question + prefix)[:self.max_seq_len]
        tokens = self.tokenizer.convert_tokens_to_ids(ques_tokenized)
        dct['ques'] = torch.Tensor(tokens).long()
        # =============</preparing question in required format>============

        # ==============<preparing answer tensors>===========
        answer_tensor = []
        for a in item['answers']:
            try:
                answer = self._extract_answer(a['answer'])
                answer_tensor.append(self.ans_vocab[answer])
            except Exception:
                continue

        one_answer_tensor = torch.Tensor([answer_tensor[0]]).long()
        answer_tensor = torch.Tensor(answer_tensor)

        dct["ans"] = one_answer_tensor
        # answerList should be of same length (same as question). For now, not
        # returning it in the dictionary.
        # dct["answerList"] = answer_tensor
        # ==============</preparing answer tensors>==========

         # inx % 4

        return dct  # [question, item['answers'], image]


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
    # dataset = VQAOne("/home/prabhu/VQA1.0/v2_mscoco_train2014_annotations_cleaned.json", "/home/prabhu/VQA1.0/v2_OpenEnded_mscoco_train2014_questions.json", "/home/prabhu/VQA1.0/train2014")

    start_time = time.time()
    ds = MultiImageVQADataset('/home/prabhu/vqa/models/cleaned.json', '/home/prabhu/CLEVR_v1.0/images/val', '/home/prabhu/Tiny/tiny-imagenet-200/test/images')
    datasetLength = len(ds)

    train_, validation_ = torch.utils.data.random_split(ds, [int(0.8*datasetLength), len(ds) - int(0.8*datasetLength)], generator=torch.Generator().manual_seed(42))
    dl_train = DataLoader(train_, batch_size=20, num_workers=2, collate_fn=arrange_batch, shuffle=True)
    dl_val = DataLoader(validation_, batch_size=20, num_workers=2, collate_fn=arrange_batch, shuffle=True)


    # dl = DataLoader(data, batch_size=20, num_workers=20, collate_fn=arrange_batch)
    for batch in tqdm(dl_train, desc=f'Iterating over dl | time: {timedelta(seconds=time.time() - start_time)}'):
        # print(batch)
        # break
        for key, val in batch.items():
            tqdm.write(f'{key}, {val.shape}')

    print(f'Time taken: {timedelta(seconds=time.time() - start_time)}')
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


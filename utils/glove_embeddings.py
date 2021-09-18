from tqdm import tqdm
import numpy as np
from math import inf
from collections import defaultdict

class Glove:
    def __init__(self, glove_file_path = '/nfs_home/janhavi2021/glove/glove.840B.300d.txt', max_words=inf):
        self.glove_file_path = glove_file_path
        self.max_words = max_words
        self.embeddings_dict = defaultdict(lambda: np.zeros((300,), dtype=np.float32))
        self._index_to_word = []
        self.get_embeddings(self.glove_file_path, self.max_words)
        self._word_to_index = defaultdict(lambda: len(self._index_to_word))
        self._word_to_index.update({word: index for index, word in enumerate(self._index_to_word)})

    def get_embeddings(self, glove_file_path = '/nfs_home/janhavi2021/glove/glove.840B.300d.txt', max_words=inf):
        with open(glove_file_path, 'r') as f:
            for index, line in tqdm(enumerate(f), desc='Reading glove....'):
                if index > max_words:
                    break
                try:
                    values = line.split()
                    word = values[0]
                    # print(word, len(values[1:]))
                    vector = np.asarray(values[1:], "float32")
                    self.embeddings_dict[word] = vector
                except Exception as e:
                    # print(e)
                    pass

        # add '[PAD]' token
        self.embeddings_dict['[PAD]'] = np.zeros((300,), 'float32')
        self.embeddings_dict['[PAD]'][0] = 1.0
        self._index_to_word = list(self.embeddings_dict.keys())
        self._index_to_word.sort()

    def word_to_index(self, word):
        return self._word_to_index[word]
    
    def index_to_word(self, index):
        if index < len(self._index_to_word):
            return self._index_to_word[index]
        return '[UNK]'

    def index_to_embeddings(self, index):
        word = self.index_to_word(index)
        return self.embeddings_dict[word]

    def __len__(self):
        return len(self._index_to_word)

# if __name__ == '__main__':
    # word_list, embeddings_dict = get_embeddings(max_words=10)
    # print(f'Total num of words: {len(word_list)}, embeddings dimention: {embeddings_dict[word_list[0]].shape}')
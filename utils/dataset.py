import torch 
import torch.nn
from torch.utils.data import Dataset, DataLoader


class MultiImageVQADataset(Dataset):
    def __init__(self):
        self.n_images = 3


    def __len__(self):
        return 10

    def __getitem__(self, idx):
        dct = {
            f'images': [torch.randint(255, size=(3, 448, 448)) / 255 for x in range(1, self.n_images+1)]
        }

        batch_size, seq_len = 1, 5
        dct['ques'] = torch.randint(1000, size=(seq_len,))
        dct['ans'] = torch.randint(9000-2, size=(1,))
        return dct

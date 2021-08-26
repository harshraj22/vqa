from torch.utils.data import Dataset
import cv2, wget 

class TextVQA(Dataset):
  def __init__(self, valJson):
    """
    valJson: path to json file containing annotations
    """
    file_ = open(valJson)

    self.jsonData = json.load(file_)
    self.listData = self.jsonData['data']

    self.length = len(self.listData)
  
  def __len__(self):
    return self.length
  
  def __getitem__(self, idx):

    item_ = self.listData[idx]

    url_ = item_['flickr_original_url']
    height = item_['image_height']
    width = item_['image_width']

    image_ = wget.download(url_)
    np_image = cv2.imread(image_)

    return item_['image_id'], item_['question_id'], item_['question'], item_['answers'], item_['question_tokens'], np_image

if __name__ == '__main__':
  data = TextVQA('TextVQA_0.5.1_val.json')
  data.__len__()

  data.__getitem__(45)

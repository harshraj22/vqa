import torch.nn as nn
from torch.nn.modules.rnn import LSTM
from transformers import BertModel
import torch 

class LanguageOnlyNetwork(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  ).eval()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(1, 1, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(372, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 1)

        self.sig = nn.Sigmoid()
   
    
    def forward(self, question):
        segments_ids = [1] * len(question[0]) # questions are only single sentences
        batched_segment_ids = [segments_ids] * len(question)
        batched_segment_ids = torch.Tensor(batched_segment_ids).long()

        with torch.no_grad():
            embeddings = self.bert(question, batched_segment_ids) # 3 32 30 768 [layers, batches, max_seq_len, embedding_size]

        features = torch.stack(embeddings[2], dim=0).sum(dim=0)

        features = features.unsqueeze(dim=1)
        
        singleFeature = self.conv1(features)
        singleFeature = self.conv2(singleFeature)
        singleFeature = self.conv3(singleFeature)
        singleFeature = self.conv4(singleFeature)
        singleFeature = self.conv5(singleFeature).flatten(start_dim=1, end_dim=3)

        f = self.fc1(singleFeature)
        f = self.fc2(f)
        f = self.fc3(f)
        f = self.fc4(f)
        f = self.fc5(f)
        out = self.sig(f) # between 0 and 1

        return out #(out*30000).int()

if __name__ == '__main__':
    question = torch.ones((32, 30)).long()

    model = LanguageOnlyNetwork(30)
    print("question is: ", question.shape)
    output = model(question)

    print(output)

# print(len(output[0][0][0]))
# print(type(output[2]))
# token_embeddings = torch.stack(output[2], dim=0)
# print(token_embeddings.shape)

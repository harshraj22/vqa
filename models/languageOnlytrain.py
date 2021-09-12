from languageOnly import LanguageOnlyNetwork
from tqdm import tqdm 
import torch.nn.functional as F
import torch 
from dataset import MultiImageVQADataset
from torch.utils.data import DataLoader
import torch.nn as nn 
from torch.autograd import Variable

epochs = 10
learningRate = 0.01
batchSize = 2048

dataset = MultiImageVQADataset('/nfs_home/janhavi2021/vqa/models/cleaned.json', '/nfs_home/janhavi2021/clever/CLEVR_v1.0/images/val', '/nfs_home/janhavi2021/Tiny/tiny-imagenet-200/test/images')
trainLoader = DataLoader(dataset, batch_size=batchSize)
print("Dataset Length. ", len(dataset))

max_seq_len = 30

model = LanguageOnlyNetwork(max_seq_len)

optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    epochLoss = 0
    with tqdm(trainLoader, unit="batch", position=0, leave=True, ascii=True) as tepoch:
                for inx, batch in enumerate(tepoch):
                    optimizer.zero_grad()
                    question, answer = batch['ques'], batch['ans']

                    outputToken = model(question)
                    outputToken = (outputToken*30000) #.int()

                    answer = answer.float()
                    answer.requires_grad = True
                    
                    loss = F.mse_loss(outputToken, answer) #(outputToken - answer)*(outputToken - answer)

                    loss.backward()
                    optimizer.step()

                    epochLoss += loss.item()

                    if inx != 0:
                        print(epochLoss/inx)

                    if inx == 10 :
                        print(question[0], outputToken[0])
    print(epochLoss)
    torch.save('languageOnly.pt', model)


# def inference(question)
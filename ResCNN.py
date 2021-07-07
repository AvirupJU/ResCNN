import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.FastConv import fastconv2d
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import math
import matplotlib.pyplot as plt 
from time import time
'''
def svd(img, rank):
    u, s, vt = torch.linalg.svd(img_batch, full_matrices=True)
    s = torch.diag_embed(s)
    u = torch.narrow(u,2,0,rank)
    s = torch.narrow(s,2,0,rank)
    s = torch.narrow(s,1,0,rank)
    vt = torch.narrow(vt,1,0,rank)
    return(u @ s @ vt)
'''
class SVDecomp:
    
    def __init__(self, rank):
        self.rank = rank

    def __call__(self, sample):
        inputs = sample
        u, s, vt = torch.linalg.svd(inputs, full_matrices=True)
        s = torch.diag_embed(s)
        u = torch.narrow(u,2,0,self.rank)
        s = torch.narrow(s,2,0,self.rank)
        s = torch.narrow(s,1,0,self.rank)
        vt = torch.narrow(vt,1,0,self.rank)
        inputs = u @ s @ vt
        return inputs

class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(CNN,self).__init__()
        self.conv1 = fastconv2d(in_channels=1,out_channels=8,kernel_size=(3,3),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = fastconv2d(in_channels=8,out_channels=16,kernel_size=(3,3),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7,num_classes)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x

rank = 10
in_channels = 1
num_classes = 10
batch_size = 128
lr_rate = 0.07
epochs = 4

composed = transforms.Compose([transforms.ToTensor(), SVDecomp(rank)])

train_dataset = datasets.MNIST(root='data/',
                               train=True,
                              transform=composed,
                              download=False)
train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='data/',
                               train=False,
                              transform=composed,
                              download=False)
test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr_rate)
t1 = time()
for epoch in range(epochs):
    for batch_idx,(data,targets) in enumerate(train):

        #data = data.reshape(data.shape[0],-1)
        
        scores = model(data)
        loss = criterion(scores,targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
    print(loss)
t2 = time()        
def check_acc(loader,model):
    if loader.dataset.train:
        print("On train data:")
    else:
        print("On test data:")
    correct,samples =0,0
    model.eval()
    
    with torch.no_grad():
        net_acc,count=0,0
        for x,y in loader:
            count+=1
            #x = x.reshape(x.shape[0],-1)
            scores = model(x)
            _,predictions = scores.max(1)
            correct+=(predictions==y).sum()
            samples+=predictions.size(0)
            
            net_acc+=float(correct)/float(samples)*100
        print("Accuracy is: ",np.round(net_acc/count,2))

print("Time taken in training loop:",(t2-t1))            
check_acc(train,model)
check_acc(test,model)

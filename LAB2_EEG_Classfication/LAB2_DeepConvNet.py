import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils

import matplotlib.pyplot as plt
import numpy as np

from dataloader import read_bci_data

train_data, train_label, test_data, test_label = read_bci_data()

CUDA = False
if torch.cuda.is_available():
    device = torch.device("cuda")
    CUDA = True

tensor_train_data = torch.stack([torch.Tensor(i) for i in train_data]) # transform to torch tensors
tensor_train_label = torch.from_numpy(train_label)

dataset = utils.TensorDataset(tensor_train_data, tensor_train_label) # create your datset
dataloader = utils.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4) # create your dataloader

tensor_test_data = torch.stack([torch.Tensor(i) for i in test_data]) # transform to torch tensors
tensor_test_label = torch.from_numpy(test_label)

test_dataset = utils.TensorDataset(tensor_test_data, tensor_test_label) # create your datset
test_dataloader = utils.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4) # create your dataloader

class DCNet_ELU(nn.Module):
    def __init__(self):
        super(DCNet_ELU, self).__init__()
        
        # Layer 1
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), bias=True),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # Layer 2
        self.secondConv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # Layer 3
        self.thirdConv = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # Layer 4
        self.forthConv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # FC Layer
        self.classify = nn.Sequential(
            nn.Linear(8600, 2)
        )
        
    def forward(self, x):
        out = self.firstConv(x)
        out = self.secondConv(out)
        out = self.thirdConv(out)
        out = self.forthConv(out)
        out = out.reshape((x.shape[0], -1))
        out = self.classify(out)
        return out

class DCNet_ReLU(nn.Module):
    def __init__(self):
        super(DCNet_ReLU, self).__init__()
        
        # Layer 1
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), bias=True),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # Layer 2
        self.secondConv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # Layer 3
        self.thirdConv = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # Layer 4
        self.forthConv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # FC Layer
        self.classify = nn.Sequential(
            nn.Linear(8600, 2)
        )
        
    def forward(self, x):
        out = self.firstConv(x)
        out = self.secondConv(out)
        out = self.thirdConv(out)
        out = self.forthConv(out)
        out = out.reshape((x.shape[0], -1))
        out = self.classify(out)
        return out

class DCNet_LeakyReLU(nn.Module):
    def __init__(self):
        super(DCNet_LeakyReLU, self).__init__()
        
        # Layer 1
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), bias=True),
            nn.BatchNorm2d(25),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # Layer 2
        self.secondConv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # Layer 3
        self.thirdConv = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # Layer 4
        self.forthConv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        
        # FC Layer
        self.classify = nn.Sequential(
            nn.Linear(8600, 2)
        )
        
    def forward(self, x):
        out = self.firstConv(x)
        out = self.secondConv(out)
        out = self.thirdConv(out)
        out = self.forthConv(out)
        out = out.reshape((x.shape[0], -1))
        out = self.classify(out)
        return out

ELU_Net = DCNet_ELU()
ReLU_Net = DCNet_ReLU()
LeakyReLU_Net = DCNet_LeakyReLU()
if CUDA:
    ELU_Net = ELU_Net.cuda()
    ReLU_Net = ReLU_Net.cuda()
    LeakyReLU_Net = LeakyReLU_Net.cuda()
criterion = nn.CrossEntropyLoss()
ELU_optimizer = optim.SGD(ELU_Net.parameters(), lr=0.01, momentum=0.5, nesterov=True)
ReLU_optimizer = optim.SGD(ReLU_Net.parameters(), lr=0.01, momentum=0.5, nesterov=True)
LeakyReLU_optimizer = optim.SGD(LeakyReLU_Net.parameters(), lr=0.01, momentum=0.5, nesterov=True)

def test_accuracy():
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            ELU_Net.eval()
            outputs = ELU_Net(inputs)
            ELU_Net.train()
            labels = labels.long()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    ELU_test_acc = 100 * correct / total
    
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = ELU_Net(inputs)
            labels = labels.long()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    ELU_train_acc = 100 * correct / total
    
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            ReLU_Net.eval()
            outputs = ReLU_Net(inputs)
            ReLU_Net.train()
            labels = labels.long()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    ReLU_test_acc = 100 * correct / total
    
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = ReLU_Net(inputs)
            labels = labels.long()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    ReLU_train_acc = 100 * correct / total
    
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            LeakyReLU_Net.eval()
            outputs = LeakyReLU_Net(inputs)
            LeakyReLU_Net.train()
            labels = labels.long()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    LeakyReLU_test_acc = 100 * correct / total
    
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = LeakyReLU_Net(inputs)
            labels = labels.long()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    LeakyReLU_train_acc = 100 * correct / total
    
    return ELU_test_acc, ELU_train_acc, ReLU_test_acc, ReLU_train_acc, LeakyReLU_test_acc, LeakyReLU_train_acc

epochs = []
ELU_test_accs = []
ELU_train_accs = []
ReLU_test_accs = []
ReLU_train_accs = []
LeakyReLU_test_accs = []
LeakyReLU_train_accs = []

for epoch in range(2000):
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(dataloader, 0):
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        ELU_optimizer.zero_grad()
        ReLU_optimizer.zero_grad()
        LeakyReLU_optimizer.zero_grad()
        
        ELU_outputs = ELU_Net(inputs)
        ReLU_outputs = ReLU_Net(inputs)
        LeakyReLU_outputs = LeakyReLU_Net(inputs)
        
        labels = labels.long()
        
        loss = criterion(ELU_outputs, labels)
        loss.backward()
        ELU_optimizer.step()
        
        loss = criterion(ReLU_outputs, labels)
        loss.backward()
        ReLU_optimizer.step()
        
        loss = criterion(LeakyReLU_outputs, labels)
        loss.backward()
        LeakyReLU_optimizer.step()
        
        running_loss += loss.item()
        
    ELU_test_acc, ELU_train_acc, ReLU_test_acc, ReLU_train_acc, LeakyReLU_test_acc, LeakyReLU_train_acc = test_accuracy()
    ELU_test_accs.append(ELU_test_acc)
    ELU_train_accs.append(ELU_train_acc)
    ReLU_test_accs.append(ReLU_test_acc)
    ReLU_train_accs.append(ReLU_train_acc)
    LeakyReLU_test_accs.append(LeakyReLU_test_acc)
    LeakyReLU_train_accs.append(LeakyReLU_train_acc)
    epochs.append(epoch)
    if epoch%10 == 9:
        print('[%d] loss: %.3f' % (epoch + 1, running_loss))
        running_loss = 0.0
print("Finished!!!")

plt.figure()
plt.plot(np.array(epochs[:]), np.array(ELU_test_accs[:]), label='ELU_Test')
plt.plot(np.array(epochs[:]), np.array(ELU_train_accs[:]), label='ELU_Train')
plt.plot(np.array(epochs[:]), np.array(ReLU_test_accs[:]), label='ReLU_Test')
plt.plot(np.array(epochs[:]), np.array(ReLU_train_accs[:]), label='ReLU_Train')
plt.plot(np.array(epochs[:]), np.array(LeakyReLU_test_accs[:]), label='LeakyReLU_Test')
plt.plot(np.array(epochs[:]), np.array(LeakyReLU_train_accs[:]), label='LeakyReLU_Train')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

print(max(ELU_test_accs))
print(max(ReLU_test_accs))
print(max(LeakyReLU_test_accs))


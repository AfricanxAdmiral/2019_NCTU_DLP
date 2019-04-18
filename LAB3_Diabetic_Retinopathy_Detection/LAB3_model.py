import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

CUDA = True if torch.cuda.is_available() else False

LOAD = True
CONT = False
EPOCH = 10
Model = './weight/pre_resnet50_82021.pkl'

from utils import progress_bar
import torchvision.models as models

from confusion import plot_confusion_matrix

from dataloader import RetinopathyLoader
import torch.utils.data as utils

train_dataset = RetinopathyLoader('./data/', 'train')
test_dataset = RetinopathyLoader('./data/', 'test')

trainloader = utils.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
testloader = utils.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()

        self.classify = nn.Linear(2048, 5)

        pretrained_model = models.__dict__['resnet{}'.format(50)](pretrained=True)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x

net_model = ResNet()

net18 = models.resnet18(pretrained=True)
net18.fc = nn.Linear(512, 5)
net50 = models.resnet50(pretrained=True)
net50.fc = nn.Linear(2048, 5)
if CUDA:
    net = net_model.cuda()

#print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_acc.append(100.*correct/total)

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            truth_y.append(targets.item())
            pred_y.append(predicted.item())

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    test_acc.append(100.*correct/total)
    
    if not LOAD:
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            torch.save(net.state_dict(), './model.pkl')
            best_acc = acc

best_acc = 0
train_acc = []
test_acc = []

classes = np.array([0, 1, 2, 3, 4])
pred_y = []
truth_y = []

if LOAD:
    print('Loading model ...')
    net.load_state_dict(torch.load(Model))
    test()
else:
    if CONT:
        print('Continue training !')
        net.load_state_dict(torch.load(Model))
    for epoch in range(EPOCH):
        train(epoch)
        test()

plot_confusion_matrix(truth_y, pred_y, classes, True)
plt.show()

#print(train_acc)
#print(test_acc)
# -*- coding: utf-8 -*-
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=20, metavar='E',
                    help='# of epochs to train (default: 20)')
parser.add_argument('-g', '--gpu', type=str, default='0', metavar='GPU',
                    help='set CUDA_VISIBLE_DEVICES (default: 0)')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import *


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128*8*8)
        x = self.classifier(x)
        return x


def train():
    # load dataset
    # ==========================
    trainloader, testloader = load_CIFAR10()
    N = len(trainloader)
    print('# of trainset: ', N)
    print('# of testset: ', len(testloader))

    # load model
    cnn = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters())
    cnn.cuda()
    criterion.cuda()

    # train
    # ==========================
    loss_history = []
    acc_history = []
    time_history = []
    for epoch in range(opt.epochs):
        loss_cum = 0.0
        acc_cum = 0.0
        time_cum = 0.0
        for i, (imgs, labels) in enumerate(trainloader):
            imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())
            cnn.zero_grad()

            start = time.time()
            outputs = cnn(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            time_cum += time.time() - start

            loss_cum += loss.data[0]
            acc = accuracy(outputs, labels.data)
            acc_cum += acc
            show_progress(epoch+1, i+1, N, loss.data[0], acc)

        print('\t mean acc: %f' % (acc_cum/N))
        loss_history.append(loss_cum/N)
        acc_history.append(acc_cum/N)
        time_history.append(time_cum)

    # test accuracy
    cnn.eval()
    correct, total = 0, 0
    for imgs, labels in testloader:
        imgs, labels = Variable(imgs.cuda()), labels.cuda()
        outputs = cnn(imgs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum()

    print('======================')
    print('epoch: %d  batch size: %d' % (opt.epochs, opt.batch_size))
    print('mean accuracy on %d test images: %f' % (total, correct/total))

    # save histories
    anp.savetxt('loss_history.csv', loss_history)
    np.savetxt('acc_history.csv', acc_history)
    np.savetxt('time_history_.csv', time_history)
    # save models
    torch.save(cnn.state_dict(), 'model.pth')


if __name__ == '__main__':
    train()


# -*- coding: utf-8 -*-
import sys
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def load_CIFAR10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(root='./CIFAR10', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size,
                                              shuffle=True, num_workers=1, pin_memory=True)
    testset = CIFAR10(root='./CIFAR10', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=opt.batch_size,
                                         shuffle=False, num_workers=1, pin_memory=True)
    return (trainloader, testloader)

def show_progress(e,b,b_total,loss,acc):
    sys.stdout.write("\r%3d: [%5d / %5d] loss: %f acc: %f" % (e,b,b_total,loss,acc))
    sys.stdout.flush()

def accuracy(out, labels):
    _, pred= torch.max(out.data, 1)
    return (pred == labels).sum() / labels.size(0)


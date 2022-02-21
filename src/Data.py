import torch
from PIL import Image
import numpy as np

class Data(torch.utils.data.Dataset):
    def __init__(self, x, labels, transform, idx=None):
        self.x = x
        self.labels = labels
        self.transform = transform
        self.idx = idx
    def __len__(self):
        'Denotes the total number of samples'
        if self.idx == None:
            return len(self.labels)
        else:
            return len(self.idx)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.idx != None:
            index = self.idx[index]
        img = self.x[index]
        label = self.labels[index]
        # img = Image.fromarray(img.numpy()) # for MNIST
        img = Image.fromarray(img)
        pos = self.transform(img)

        return pos, label
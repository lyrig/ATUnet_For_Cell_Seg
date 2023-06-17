import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader

from DataLoader import ImageDataset, show_data

model = torch.load('./model.pt')     
train_dir = './data/20x_segment_train_dataset/dsb2018p/train'
test_dir = './data/20x_segment_train_dataset/dsb2018p/test'
val_dir = './val.txt'
#set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

print(model)

train_dataset = ImageDataset(train_dir)
test_dataset = ImageDataset(test_dir)


plt.figure(figsize=(20, 10))
train = DataLoader(dataset=train_dataset, batch_size=5, shuffle=False)
for i, data in enumerate(train):
    (x, y) = data
    with torch.no_grad():
        y1 = model(x)
        for i in range(5):
            print(type(x.numpy()))
            print(x.numpy()[0, 0].shape)
            
            plt.subplot(3, 5, (i + 1))
            plt.imshow(x.numpy()[i, 0])
            plt.subplot(3, 5, (i + 1 + 5))
            plt.imshow(y.numpy()[i, 0])
            plt.subplot(3, 5, (i + 1 + 5 + 5))
            y1 = y1.numpy()[i, 0]
            y1[y1>0.1]=1
            plt.imshow(y1)
    break
plt.savefig('./result.png')


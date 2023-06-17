import matplotlib.pyplot as plt
import numpy as  np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from eloss import losstest, SoftDiceLoss
from DataLoader import test
from torch.utils.data import DataLoader
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
line_style = ['-', '--', '-.']

'''


'''
def train(model, dataset, lr, epochs, batch_size, loss_func, SaveEach:bool = False):
    print("Begin trainning...")
    optimizer = optim.Adam(model.parameters(), lr = lr)
    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    print("Loading model to GPU...")
    history_t = []
    history_v = []
    tloss = 0
    #dataset = DataLoader(dataset=Full_dataset, batch_size=batch_size, shuffle=False)
    (train, test) = dataset
    train = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
    #print("Begin trainning...")
    test = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    #print("Begin trainning...")
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        train_step = 0
        for i, data in enumerate(train):
            (input, target) = data
            '''
            # 如果GPU可用，将模型和张量加载到GPU上
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            '''
            output = model(input)
            #print(output)
            target = torch.sigmoid(target)
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step = train_step + 1
            if train_step % 10 == 0:
                print("Train_step: {}, loss: {}".format(train_step, loss.item()))
        print("Epoch:{}, Loss:".format(epoch))
        history_t.append(tloss)
        # Testing Step start.
        total_test_loss = 0
        
        with torch.no_grad():
            for i, data in enumerate(test):
                (input, targets) = data
                targets = torch.sigmoid(targets)
                outputs = model(input)
                loss = loss_func(outputs, targets)
                if i % 10 == 0:
                    print("Validation Step {} : {}" .format(i, loss.item()))
                total_test_loss = total_test_loss + loss.item()
        print("Total Test Loss: {}".format(total_test_loss))
        try:
            if total_test_loss > history_v[-1]:
                lr *= 0.1
                optimizer = optim.Adam(model.parameters(), lr = lr)
        except:
            pass
        history_v.append(total_test_loss)

        '''
        Consider the model is too big to save so much version.
        '''
        if SaveEach:
            torch.save(test, "test_{}.pth".format(i))
            print("The model has been saved.")
        
    return 1




def validation(model, val):
    loss_func = SoftDiceLoss(1)
    total_loss = 0
    val = DataLoader(val, batch_size=16, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(val):
            (input, target) = data
            output = model(input)
            loss = loss_func(output, target)
            total_loss = total_loss + loss.item()
        print("Validation Loss: {}".format(total_loss))


'''
Name: draw_chart
Function: Using the data and its name(type) to draw a chart.

para:
    data(list):
        include [(x1, y1) , (x2, y2), ...]
        the x1 y1 are list too.
    
    type(list):
        include the name of (x1, y1)...
        [name1, name2, ...]

    savefig(bool):
        default = True. Decide whether to save the chart as file.
'''
def draw_chart(data:list, type:list, savefig:bool=True, save_dir:str = './result.png'):
    lend = len(data)
    lent = len(type)
    try:
        assert lend == lent
    except:
        print("Draw chart Failure. The size of data{}, bu the size of type{}".format(lend, lent))
        exit(1)
    
    plt.figure(figsize=(20, 10), dpi=100)
    for i in range(lend):
        (x, y) = data[i]
        name = type[i]
        plt.plot(x, y, color[i], marker='*', markersize=2)

    plt.legend(type)
    plt.show()
    if savefig:
        plt.savefig(save_dir)
    else:
        pass
    
    return;


'''
Name: normalization
Function: Use the formule x = (x - Min)/(Max - Min), to get the data to [0, 1]

para:
    input(list):
        the input

return:
    the result(input(list))
'''
def normalization(input:list):
    maxVal = max(input)
    minVal = min(input)
    input = (input - minVal)/(maxVal - minVal)
    return input
'''
Name: chart_two
Function: To draw two data in the same chart. (Will normalize)

para:
    data1(tuple):
        input data1, the tuple include (x1:list, y1:list)
    data2(tuple):
        same as data1
    name(tuple):
        include the name of data1 and data2. like (name1, name2)
    savefig(bool): default = True
        To determine whether you want to save the chart as a png image.
    
'''
def chart_two(data1:tuple, data2:tuple, name:tuple, savefig:bool = True):
    (x1, y1) = data1
    (x2, y2) = data2
    (name1, name2) = name
    x1 = normalization(x1)
    y1 = normalization(y1)
    x2 = normalization(x2)
    y2 = normalization(y2)
    plt.figure(figsize=(20, 10))
    plt.plot(x1, y1, 'r', '*')
    plt.plot(x2, y2, 'y', '*')
    plt.show()


if __name__ == "__main__":
    losstest()
    test()


        
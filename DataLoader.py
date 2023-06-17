import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from re import findall
from os import listdir
import os
from cv2 import resize

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
class ImageDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, josn_dir, input_name:str='images', output_name:str='masks'):
        self.dataset = load_data(josn_dir, input_name, output_name)
        self.length = len(self.dataset['image'])
        assert len(self.dataset['image']) == len(self.dataset['GT'])

    def __getitem__(self, index):

        img, target = self.dataset['image'][index], self.dataset['GT'][index]
        
        return img, target

    def __len__(self):
        return self.length

'''
Name: load_data

Function: Load images from the path given by json_dir.

parameter:

    -json_dir: str
    The loading path. The content can be a filename or a foldername.
    The format must be like the following examples(don't keep the final "/"):
        './tmp1/test.txt', './train'

output:
    return a dictionary with the content:
        {"image":[...], "GT":[...] }

Note:
    This means if you want to construct train and test dataset, you need to
    use twice seperately.
'''
def load_data(json_dir:str, inputname:str = "images", outputname:str = "masks"):
    x_train = []
    y_train = []
    result = {}
    
    checking = json_dir.split('/')

    #debug
    #print(checking[-1])
    #debug

    find = findall("\.txt", checking[-1])
    #print(find[-1])
    if len(find) != 0:
        with open(json_dir, encoding='gbk') as f:
            line = f.readline().strip()
            while line:
                patch = line.split(' ')
                input_image = plt.imread('.'+patch[0])
                output_image = plt.imread('.'+patch[1])
                shape_in = input_image.shape
                shape_out = output_image.shape

                if len(shape_in) == 2:
                    '''
                    This image's shape is like (x, y), and it is a gray image.
                    To train the neural network, reshape it to (channel(3), x, y)
                    '''
                    input_image = input_image.reshape((1, shape_in[0], shape_in[1]))
                    
                    print(input_image)
                    input_image = np.concatenate((input_image, input_image, input_image), axis=0)
                    output_image = output_image.reshape((1, shape_out[0], shape_out[1]))
                    #output_image = np.concatenate((output_image, output_image, output_image), axis=0)
                
                
                input_image = torch.FloatTensor(input_image)
                output_image = filter(lambda x: 1 if x > 0 else 0, output_image)
                print(output_image)
                output_image = torch.FloatTensor(output_image)
                

                x_train.append(input_image)
                y_train.append(output_image)
                line = f.readline().strip()
    else:
        inputdir = json_dir+'/'+inputname
        outputdir = json_dir + '/' + outputname
        filelist = listdir(inputdir)
        for name in filelist:
            input_image = plt.imread(inputdir + '/' + name).astype('float16')
            output_image = plt.imread(outputdir + '/' + name).astype('float16')

            shape_in = input_image.shape
            shape_out = output_image.shape
            assert shape_in == shape_out
            if len(shape_in) == 2:
                '''
                This image's shape is like (x, y), and it is a gray image.
                To train the neural network, reshape it to (channel(3), x, y)
                '''
                input_image = input_image.reshape((1, shape_in[0], shape_in[1]))
                input_image = np.concatenate((input_image, input_image, input_image), axis=0)
                output_image = output_image.reshape((1, shape_out[0], shape_out[1]))
                #output_image = np.concatenate((output_image, output_image, output_image), axis=0)
            input_image = torch.FloatTensor(input_image)
            output_image = torch.FloatTensor(output_image)
            if torch.cuda.is_available():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                input_image = input_image.to(device)
                output_image = output_image.to(device)
            x_train.append(input_image)
            y_train.append(output_image)
    result['image'] = x_train
    result['GT'] = y_train
    return result

def show_data(dataset, name='NULL', num:int=5):
    plt.figure(figsize=(10,5)) #设置窗口大小
    plt.suptitle(name) # 图片名称
    for i in range(1, num+1):
        plt.subplot(2,num,i)
        img, target = dataset[i]
        #print(img.shape)
        if img.shape[0] <= 3:
            #print(img.shape)
            #print(type(img))
            img = img.transpose(0, 1)
            img = img.transpose(1, 2)
            #img = img.astype('int')
            target = target.transpose(0, 1)
            target = target.transpose(1, 2)
            #target = target.astype('int')
        print(img.shape==target.shape)
        plt.imshow(img), plt.axis('off')
        plt.subplot(2, num, i + num)
        plt.imshow(target), plt.axis('off')
    plt.show()
    '''
    If the server uses Linux, can use the following code to print a image to see the result.
    '''
    plt.savefig(name+'.png')

def test():
    print("DataLoader file import successful!")

if __name__ == '__main__':
    print("=========Running File DataLoader.py=========")
    test_txt = './val.txt'
    test_folder = './data/20x_segment_train_dataset/dsb2018p/train'
    dataset = ImageDataset(test_folder, 'images', 'masks')
    raw, GT = dataset[0]
    print(raw)
    #dataset = torch.FloatTensor(dataset)
    #show_data(dataset)
    #print("OKOKKKKKKK")
    tmp = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    #for i, data in enumerate(tmp):
        #print('No.'+str(i))
        #t1, t2 = data
        #print(len(t1))
    print("==========Ending File DataLoader.py=========")


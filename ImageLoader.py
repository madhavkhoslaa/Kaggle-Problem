from torch.utils.data import Dataset
import glob
from random import shuffle
import skimage
from skimage import transform
from torchvision import transforms
import torch


class Image_Loader():
    def __init__(self, class1, class2,train_percent):
        self.Image=[]
        self.class1_images= glob.glob(class1)
        self.class2_images= glob.glob(class2)
        self.Image= self.class1_images+ self.class2_images
        self.train_percent= train_percent
        shuffle(self.Image)
        train_len= int(train_percent*len(self.Image))
        self.train_set= self.Image[0:train_len]
        self.test_set= self.Image[train_len:]

class Cell_Images(Dataset):
    def __init__(self, Image_list):
        self.Images= Image_list
    def __getitem__(self, index):
        img_= self.Images[index]
        if "Parasitized" in img_:
            label= 0
        else:
            label= 1
        image_ = transform.resize((skimage.io.imread(img_)), (256, 256, 3))
        sample= {"Image": image_[:3], "Label": label}

        return sample
    def __len__(self):
        return len(self.Images)

class Cell_Test_Images(Dataset):
    def __init__(self, Image_list):
        self.Images= Image_list
    def __getitem__(self, index):
        img_= self.Images[index]
        if "Parasitized" in img_:
            label= 0
        else:
            label= 1
        image_ = transform.resize((skimage.io.imread(img_)), (256, 256, 3))
        sample= {"Image": image_[:3], "Label": label}
        return sample
    def __len__(self):
        return len(self.Images)

from model import model
from ImageLoader import Cell_Images
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import skimage
from ImageLoader import Image_Loader
Infected="/home/fatjuicyboi/DataSets/cell-malayria/cell_images/Parasitized/*png"
Uninfected="/home/fatjuicyboi/DataSets/cell-malayria/cell_images/Uninfected/*png"
loader= Image_Loader(Infected, Uninfected, 0.7)
net= model().cuda()
dset_train= Cell_Images(loader.train_set)
dset_test= Cell_Images(loader.test_set)
dataloader_train= DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=1)
dataloader_test= DataLoader(dset_test, batch_size=1, shuffle=True, num_workers=1) 
loss_function= nn.CrossEntropyLoss()
optim= torch.optim.SGD(net.parameters(), lr= 0.001, momentum= 0.05)

for epoch in range(1, 200):
    print("Started")
    running_loss= 0.0
    for i, data in enumerate(dataloader_train, 0):
        Image= data["Image"].cuda()
        Label= data["Label"].cuda()
        optim.zero_grad()
        Image = Image.type(torch.cuda.FloatTensor)
        output= net(Image)
        loss= loss_function(output, Label)
        running_loss+= loss
        loss.backward()
        optim.step()
        if(epoch%10==0 or epoch==1):
            print("Epoch Loss:", loss.item())
            print("Epoch Number:", epoch)
        if(epoch%10==0):
            for i, data in enumerate(dataloader_test, 0):
                Image= data["Image"].cuda()
                Label= data["Label"].cuda()
                Image= Image.type(torch.cuda.FloatTensor)
                net.eval()
                yhat = net(Image)
                val_loss = loss_function(yhat, Label)
                print(val_loss.item())

print(net.state_dict())

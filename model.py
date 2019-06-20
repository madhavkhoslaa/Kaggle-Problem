import torch.nn as nn
import torch.nn.functional as F
import torch
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1= nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1)
        self.pool1= nn.AvgPool2d(kernel_size=2)
        self.conv2= nn.Conv2d(in_channels=9, out_channels=3, kernel_size= 1)
        self.fc1= nn.Linear(189, 120)
        self.fc2= nn.Linear(120, 90)
        self.fc3= nn.Linear(90, 60)
        self.fc4= nn.Linear(60, 40)
        self.fc5= nn.Linear(40, 20)
        self.fc6= nn.Linear(20, 10)
        self.fc7= nn.Linear(10, 5)
        self.fc8= nn.Linear(5, 2)
    def forward(self, x):
        x= self.conv1(x)
        x= self.pool1(x)
        x= self.conv2(x)
        x= self.pool1(x)
        x= x.view(-1, 189)
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= F.relu(self.fc3(x))
        x= F.relu(self.fc4(x))
        x= F.relu(self.fc5(x))
        x= F.relu(self.fc6(x))
        x= F.relu(self.fc7(x))
        x= torch.sigmoid(self.fc8(x))
        return x
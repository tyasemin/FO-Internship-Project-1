import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size, n_classes):
        super(CNNmodel, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1,padding=0)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1,padding=0)
        self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=1)

        self.fc1=nn.Linear(n_classes*) #other variables?


    def forward(self,x):
        x=self.conv1(x)
        x=torch.relu(x)
        x=self.maxpool(x)
        
        x=self.conv2(x)
        x=nn.Softmax(dim=1)(x)
        x=self.maxpool2(x)
        x=out.view(x.size(0),-1)
        x=self.fc1(x)

        return x

if __name__ == '__main__':
    model = CNN(input_size=(224, 224), n_classes=2)

import torch.nn as nn

class ConvNet(nn.Module): 
    def __init__(self, num_classes=10): 
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential( nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2), 
        nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.layer2 = nn.Sequential( nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2), 
        nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)) 
        self.drop_out = nn.Dropout() 
        self.fc1 = nn.Linear(8 * 8 * 192, 1000) 
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x): 
        out = self.layer1(x) 
        out = self.layer2(out) 
        out = out.reshape(out.size(0), -1) 
        out = self.drop_out(out) 
        out = self.fc1(out) 
        out = self.fc2(out) 
        return out

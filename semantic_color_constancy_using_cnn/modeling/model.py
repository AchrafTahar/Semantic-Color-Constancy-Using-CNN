import torch
import torch.nn as nn
import torch.nn.functional as F

class TrueColorNet(nn.Module):
    def __init__(self):
        super(TrueColorNet, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(4, 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 2048)
        self.fc8 = nn.Linear(2048, 1024)
        self.fc9 = nn.Linear(1024, 4)  # r, g, b, gamma
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Normalization layers
        self.norm = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0)
        
    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        # Conv3-5
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected with ReLU and dropout
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        x = F.relu(self.fc8(x))
        x = self.dropout(x)
        x = self.fc9(x)
        
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    print("Testing TrueColorNet...")
    x = torch.randn()
    model = TrueColorNet()
    y = model(x)
    print(y.detach().shape)

import torch
import torch.nn as nn 
from config import *


model = nn.Sequential(
    #layer1
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), # Output size= 16 x224 x224
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels =64,kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),# Output size= 64 x 112 x 112

    #Layer 2
    nn.Conv2d(in_channels= 64, out_channels=128,kernel_size=3, stride=1, padding=1), 
    nn.ReLU(),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
    nn.ReLU(),
    nn.MaxPool2d(2,2), # Output size= 128 x 56 x 56

    #Layer 3
    nn.Conv2d(in_channels= 128, out_channels=256,kernel_size=3, stride=1, padding=1), 
    nn.ReLU(),
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), 
    nn.ReLU(),
    nn.MaxPool2d(2,2), # Output size= 256 x 28 x 28

    # Fully connected layer 
    nn.Flatten(),
    nn.Linear(256*28*28, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 8),

)


model = nn.Sequential(
    #layer1
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), # Output size= 16 x224 x224
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels =64,kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),# Output size= 64 x 112 x 112

    #Layer 2
    nn.Conv2d(in_channels= 64, out_channels=128,kernel_size=3, stride=1, padding=1), 
    nn.ReLU(),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
    nn.ReLU(),
    nn.MaxPool2d(2,2), # Output size= 128 x 56 x 56

    #Layer 3
    nn.Conv2d(in_channels= 128, out_channels=256,kernel_size=3, stride=1, padding=1), 
    nn.ReLU(),
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), 
    nn.ReLU(),
    nn.MaxPool2d(2,2), # Output size= 256 x 28 x 28

    # Fully connected layer 
    nn.Flatten(),
    nn.Linear(256*28*28, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 8),

)

model.to(device)
optim =torch.optim.Adam(model.parameters(), lr=0.001)


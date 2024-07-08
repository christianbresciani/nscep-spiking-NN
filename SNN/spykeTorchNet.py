
import SpykeTorch.neurons as neurons
import torch.nn as nn
import torch
import numpy as np

from network import CustomLoss, Net

class MyNetwork(Net):
    def __init__(self, num_step, num_classes, threshold=1.0, device = None):
        super(MyNetwork, self).__init__(device=device)
        
        self.num_step = num_step

        # self.fuse_antennas = nn.Conv2d(4, 1, (1,1))
        self.conv1 = nn.Conv1d(4, 8, 5) # [b, 4, 2048] -> [b, 8, 2044]
        self.pool1 = nn.MaxPool1d(4) # [b, 8, 2044] -> [b, 8, 511]
        self.conv2 = nn.Conv1d(8, 4, 12) # [b, 8, 511] -> [b, 4, 500]
        self.pool2 = nn.MaxPool1d(4)    # [b, 4, 500] -> [b, 4, 125]
        self.flatten = nn.Flatten(start_dim=0)     # [b, 4, 125] -> [b, 500]
        self.neuron1 = neurons.EIF(threshold, theta_rh=threshold*.8)

        self.linear1 = nn.Linear(500, 20)
        self.neuron2 = neurons.EIF(threshold, theta_rh=threshold*.8)

        self.linear2 = nn.Linear(20, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss = CustomLoss(num_outputs=num_classes, device=device)


    def forward(self, input):
        out = []
        if len(input.shape) < 4:
            input = input.unsqueeze(0)

        input = input.view(input.shape[0], -1, self.num_step, input.shape[3], input.shape[2])
        for b in range(input.shape[0]):
            self.neuron1.reset()
            self.neuron2.reset()
            for inter in range(input.shape[1]):
                for step in range(self.num_step):
                    x = input[b, inter, step, :]
                    x = self.conv1(x)
                    x = self.pool1(x)
                    x = self.conv2(x)
                    x = self.pool2(x)
                    x = self.flatten(x)
                    x = self.neuron1(x, return_winners=False)[0]

                x = self.linear1(x)
                x = self.neuron2(x, return_winners=False)[0]
                
            out.append(x)
                
        x = self.linear2(torch.stack(out))
        return self.softmax(x)
            
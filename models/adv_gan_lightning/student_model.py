import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()

        self.input_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.output_net = nn.Sequential(
            nn.Linear(64 * 4 * 4, 200),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = self.input_net(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.output_net(x)
        return x

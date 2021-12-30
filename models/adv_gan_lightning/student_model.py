import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()

        self.input_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.output_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.input_net(x)
        x = x.permute(0, 2, 3, 1)  # CRUCIAL MAGIC FOR TF-compatibility
        x = self.output_net(x)
        return x


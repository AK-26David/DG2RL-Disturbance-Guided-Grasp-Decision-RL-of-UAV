import torch
import torch.nn as nn

class DG2GraspNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
        )

        self.q_head = nn.Conv2d(32, 1, 1)
        self.cos_head = nn.Conv2d(32, 1, 1)
        self.sin_head = nn.Conv2d(32, 1, 1)
        self.width_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        q = torch.sigmoid(self.q_head(x))
        cos = torch.tanh(self.cos_head(x))
        sin = torch.tanh(self.sin_head(x))
        width = torch.sigmoid(self.width_head(x))

        return q, cos, sin, width
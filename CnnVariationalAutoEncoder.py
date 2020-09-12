import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    @staticmethod
    def reParameterize(mu, logVar):
        std = logVar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu.to(device) + std.to(device) * esp.to(device)
        return z.to(device)

    def bottleneck(self, h):
        mu, logVar = self.fc1(h), self.fc2(h)
        z = self.reParameterize(mu, logVar)
        return z, mu, logVar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logVar = self.bottleneck(h)
        return z, mu, logVar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logVar = self.encode(x)
        z = self.decode(z)
        return z, mu, logVar

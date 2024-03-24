import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the encoder neural network spr
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc21 = nn.Linear(512, 128)
        self.fc22 = nn.Linear(512, 128)

    def forward(self, x):
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(x1))
        x3 = nn.functional.relu(self.conv3(x2))
        x4 = nn.functional.relu(self.conv4(x3))
        x5 = x4.view(-1, 256 * 8 * 8)
        x6 = nn.functional.relu(self.fc1(x5))
        mu = self.fc21(x6)
        logvar = self.fc22(x6)
        return mu, logvar,(x1,x2,x3)

# Define the decoder neural network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256 * 8 * 8)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z, encoder_outputs):
        z = nn.functional.relu(self.fc1(z))
        z = nn.functional.relu(self.fc2(z))
        z = z.view(-1, 256, 8, 8)
        z = nn.functional.relu(self.conv1(z) + encoder_outputs[2])  # skip connection
        z = nn.functional.relu(self.conv2(z) + encoder_outputs[1])  # skip connection
        z = nn.functional.relu(self.conv3(z) + encoder_outputs[0])  # skip connection
        z = self.conv4(z)
        z = torch.sigmoid(z)
        return z


# Define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, encoder_outputs = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, encoder_outputs)
        return recon_x, mu, logvar

# Define the loss function
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # see the Beta-VAE paper for the derivation of this KL divergence term
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

# Define the optimizer
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

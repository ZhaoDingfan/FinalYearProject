import numpy as np
import torch
import argparse
import torch.utils.data

from Vae_Input import VAEInput

from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='VAE Implementation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cpu")
# Once GPU is available it will use GPU to train the model
# device = torch.device("cuda")

# load data 
train_loader = torch.utils.data.DataLoader(
    VAEInput('/Users/dingfan/FinalYearProject/VAE/Data/', 'train', transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    VAEInput('/Users/dingfan/FinalYearProject/VAE/Data/', 'test', transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True
)

class VAE(nn.Module):

    # Initialize all the vectors 
    def __init__(self, ):
        super(VAE, self).__init__()

        # Encode 28 * 28 vector to 1 * 400 vector
        self.fc1 = nn.Linear(784, 400)

        # Encode to latent vector 
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        # decode process
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    # Encode to the hidden layer to get the latent model parameter, mu and esilon
    def encode(self, x):
        h1 = F.relu(self.fc1(x)) # rectified linear unit
        return self.fc21(h1), self.fc22(h1)

    # Calculate the latent vector, move the random sampling out of the layer to faciliate back propogation
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # Decode latent vector back to vector that are similar to input 
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    # Model moving forward process
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3) # lr stands for learning rate

# loss function for basic Variational Autoencoder
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Model training process
def train(epoch):
    model.train()
    train_loss = 0
    for batch_index, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        # log after one log_interval to check the detailed loss information 
        if batch_index % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                100. * batch_index / len(train_loader),
                loss.item() / len(data)))

    # log after each epoch
    print('==> Epoch: {} Average Loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)
    ))

# throw the encoder part and go directly to the decoder part
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), '/Users/dingfan/FinalYearProject/VAE/Results/recon_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('==> Test set loss: {:.4f}'.format(test_loss))

# Going through for multiple epoches to improve the performance
for epoch in range(1, args.epochs+1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28), '/Users/dingfan/FinalYearProject/VAE/Results/sample_' + str(epoch) + '.png')





        







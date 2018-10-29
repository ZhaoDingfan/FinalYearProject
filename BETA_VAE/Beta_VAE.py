# generate the latent traversal and implement the model
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

# Try to generate latent traversal
# The latent traversal does not perform quite good on cat dataset although it performs well in the dot dataset. 
# Hence I would like to try another implementation to find the disentanglement of cat. 

# The preparation process is similar to normal VAE
parser = argparse.ArgumentParser(description='VAE Implementation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--beta', default=8, type=float, 
                    help='beta parameter for KL-term in original beta-VAE')
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


class BetaVAE(nn.Module):
    def __init__(self):
        super(BetaVAE, self).__init__()

        # Encode 28 * 28 vector to 1 * 400 vector
        self.fc1 = nn.Linear(784, 400)

        # Encode to latent vector 
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        # decode process
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.latent_vector = torch.zeros(args.batch_size, 20)
        print(self.latent_vector.size())

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.rand_like(std)
            return eps.mul(std).add_(mu)
        else:
            """ test mode """
            return mu
            
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        self.latent_vector = z
        return self.decode(z), mu, logvar


model = BetaVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar, beta):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_index, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, args.beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_index % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                100. * batch_index / len(train_loader),
                loss.item() / len(data)))
        
    print('==> Epoch: {} Average Loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)
    ))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar, args.beta).item()
            if i == 0:
                n = min(data.size(0), 8)
                # default batch size 128
                # comparison = torch.cat(([data[:n]], recon_batch.view(args.batch_size, 1, 28, 28)))
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 
                    '/Users/dingfan/FinalYearProject/BETA_VAE/Results/recon_' + str(epoch) + '.png',
                nrow= n)
    
    test_loss /= len(test_loader.dataset)
    print('==> Test set loss: {:.4f}'.format(test_loss))

# function to generate the latent traversal
def generate_latent_traversal(latent):
    
    sample = torch.randn(160, 20)
    for dimension in range(20):
        val = -3
        
        for row in range(8):
            new_row = latent[row].clone()
            new_row[dimension] = val
            sample[dimension*8+row] = new_row.clone()
            val += 6/7
   
    sample = model.decode(sample).cpu()

    save_image(sample.view(160, 1, 28, 28), 
        '/Users/dingfan/FinalYearProject/BETA_VAE/Results/traversal_total_' + str(args.beta) + '.png')

for epoch in range(1, args.epochs+1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        
        save_image(sample.view(64, 1, 28, 28), 
            '/Users/dingfan/FinalYearProject/BETA_VAE/Results/sample_' + str(epoch) + '.png')
    
generate_latent_traversal(model.latent_vector)




        

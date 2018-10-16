# This class will share the same input with basic VAE
# The CVAE model is similar to VAE, the only difference is that it will add some classifiers about the model input
# to constrain the reconstructed output images. The images will be constrained to a single category like cat or fish

# Thanks for the help from Zhuohan

import os
import numpy as np
import torch
import argparse
import torch.nn.functional as F

from torch import nn
from torch import optim
from sklearn.preprocessing import LabelBinarizer
from torchvision import transforms
from Vae_Input import VAEInput
from label import labels 

label_raw = labels
print(label_raw)

device = torch.device("cpu")

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

# It shares the same trainer loader and test loader with basic VAE
train_loader = torch.utils.data.DataLoader(
    VAEInput('/Users/dingfan/FinalYearProject/VAE/Data/', 'train', transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    VAEInput('/Users/dingfan/FinalYearProject/VAE/Data/', 'test', transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True
)

class CVAE(nn.Module):
    # Initialise the class
    # This round of implementation will make it more parameterized 
    def __init__(self, ):
        super(CVAE, self).__init__()
        self.feature_size = 784
        self.hidden_layer_size = 400
        self.latent_size = 20

        # Every category has one kind of class_size for distinguishment. 
        # We will use one-hot encoding 
        self.class_size = 10

        # input layer
        self.fc1 = nn.Linear(self.feature_size + self.class_size, self.hidden_layer_size)

        # encode latent layer
        # init two parellel hidden layers to represent the average and var of the latent vector
        self.fc21 = nn.Linear(self.hidden_layer_size, self.latent_size)
        self.fc22 = nn.Linear(self.hidden_layer_size, self.latent_size)

        # decode latent layer
        self.fc3 = nn.Linear(self.latent_size + self.class_size, self.hidden_layer_size)

        # decide layer
        self.fc4 = nn.Linear(self.hidden_layer_size, self.feature_size)

        self.lb = LabelBinarizer()

    # encode the category using one hot encoding to make it as the same class size with other categories 
    # while still being unique
    def convert_category(self, c: torch.FloatTensor):
        c_n = c.numpy()
        self.lb.fit(list(range(0, self.class_size)))

        c_one_hot = self.lb.transform(c_n)
        floatTensor = torch.FloatTensor(c_one_hot)
        return floatTensor
        
    def encode(self, x, c):
        category = self.convert_category(c)

        # concatenate one hot encoding with input feature size
        con = torch.cat((x, category), 1)

        # The following process will be similar to VAE model
        h1 = F.relu(self.fc1(con))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z, c):
        category = self.convert_category(c)

        # When we decode, we also have to concatenate latent vector with category
        con = torch.cat((z, category), 1)
        h3 = F.relu(self.fc3(con))
        return F.sigmoid(self.fc4(h3))
    
    def reparameterize(self, mu, logvar):
        # If in training mode, it will enable the reparameterization 
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else: 
            # If in test mode, it will simply return the mu
            return mu

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.feature_size), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3) # lr stands for learning rate

def train():
    # Function to train my model
    model.train()
    train_loss = 0
    for batch_index, (data, label)  in enumerate(train_loader):
        data = data.to(device)

        # It will also attach the one-hot label to the end of original data 
        label = model.convert_category(data)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(label)
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

def test():
    # Function to test my model
    print("test")

def loss_function():
    print("loss")

for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
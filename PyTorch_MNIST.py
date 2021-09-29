import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mnist import MNIST

# Get cpu or gpu device for training.
device = "cpu"
path = "ClassicMNIST"

#mndata = MNIST(path)
#train = mndata.load_training()
#test = mndata.load_testing()

##train = reader.load_mnist(path, 'train')
##test = reader.load_mnist(path, 'test')
##
##trans = transforms.Compose([transforms.ToTensor()])
##
##train_data = trans(train)
##test_data = trans(test)

training_data = datasets.MNIST(
    root=path,
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root=path,
    train=False,
    download=False,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

##labels_map = {
##    0: "T-Shirt",
##    1: "Trouser",
##    2: "Pullover",
##    3: "Dress",
##    4: "Coat",
##    5: "Sandal",
##    6: "Shirt",
##    7: "Sneaker",
##    8: "Bag",
##    9: "Ankle Boot",
##}

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        #self.dropout = nn.Dropout(p=0.25)
        self.linear_relu_stack = nn.Sequential(
            #nn.Dropout(p=0.3),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        

    def forward(self, x):
        x = self.flatten(x)
        #x = self.dropout(x)
        x = self.linear_relu_stack(x)
        return x

def train_loop(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if batch %100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"\nTest Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

learning_rate = 3e-3
batch_size = 64
epochs = 10

model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------")
    train_loop(train_dataloader, model, loss_fn, optimiser)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("\n\nSaving model...")
torch.save(model.state_dict(), "ClassicMNISTModel.pth")

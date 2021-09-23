import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from mnist import MNIST
import mnist_reader as reader

# Get cpu or gpu device for training.
device = "cpu"
path = "data"

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

training_data = datasets.FashionMNIST(
    root=path,
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root=path,
    train=False,
    download=False,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


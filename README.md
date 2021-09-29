# Experiments with MNIST

This repo documents my experiments and practise of machine learning using the MNIST dataset. The classic and Zalando MNIST datasets are both present here. The best model for each approach is saved in the repo home directory.

## Best Results:

### Classic MNIST:

| Method: | Test Error |
| ------- | ---------- |
| SK Learn MLP Regressor | 3.39730 |
| PyTorch | 90.5% |

### SKLearn MLP Regressor Settings:

hidden_layer_sizes=(50,1, 100,2, 20,3),
activation='identity', batch_size=3000,                                    learning_rate_init=0.02,
random_state=0,
max_iter=500


### PyTorch Settings:

Error model: Cross-Entropy Loss
ANN Structure: Linear ReLU Stack

>        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

### Fashion MNIST: 
(tbc)

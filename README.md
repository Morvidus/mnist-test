# Experiments with MNIST

This repo documents my experiments and practise of machine learning using the MNIST dataset. The classic and Zalando MNIST datasets are both present here.

## Best Results:

### Classic MNIST:

| Method: | Train Error | Test Error |
| ------- | ----------- | ---------- |
| SK Learn MLP Regressor | 3.47995 | 3.39730 |
| PyTorch | - | - |

### SKLearn MLP Regressor Settings:

hidden_layer_sizes=(50,1, 100,2, 20,3),
activation='identity', batch_size=3000,                                    learning_rate_init=0.02,
random_state=0,
max_iter=500


### PyTorch Settings:

(TBC)

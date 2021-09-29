# MNIST Test
from mnist import MNIST
from sklearn.metrics import mean_squared_error
import sklearn.neural_network as neural_network
import random
import numpy as np
import pandas as pd
import time

mndata = MNIST('./FashionMNIST/')

train_X, train_Y = mndata.load_training()
test_X, test_Y = mndata.load_testing()
train_Y = np.array(train_Y)
test_Y = np.array(test_Y)

model = neural_network.MLPRegressor(hidden_layer_sizes=(50,1, 100,2, 20,3),
                                    activation='identity', batch_size=3000,
                                    learning_rate_init=0.02, random_state=0,
                                    max_iter=500)

start_time = time.time()
model.fit(train_X, train_Y)
end_time = time.time()

Ypred = model.predict(train_X).astype(int)

print("Train Error: {:.5f}\n".format(mean_squared_error(train_Y, Ypred)))
print("Training time: %s seconds\n" % (end_time-start_time))

pred_y = model.predict(test_X).astype(int)

print("Test Error: {:.5f}\n".format(mean_squared_error(test_Y, pred_y)))

output = pd.DataFrame({'Actual': train_Y, 'Train Prediction': Ypred})
output2 = pd.DataFrame({'Test': np.array(test_Y), 'Test Prediction': pred_y})
output.to_csv('train_out.csv', index=False)
output.to_csv('test_out.csv', index=False)

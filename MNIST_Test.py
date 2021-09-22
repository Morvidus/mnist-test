# MNIST Test
from mnist import MNIST
import sklearn.neural_network as neural_network
import random
import numpy as np
import pandas as pd
import time

# Error function
def error(actual, predicted):
    return np.sqrt(np.sum(np.square(actual-predicted)/len(actual)))

mndata = MNIST('./')

X, Y = mndata.load_training()
test_X, test_Y = mndata.load_testing()
Y = np.array(Y)

model = neural_network.MLPRegressor(hidden_layer_sizes=(50,1, 100,2, 50,3),
                                    activation='identity', batch_size=2000,
                                    learning_rate_init=0.03, random_state=0,
                                    max_iter=500)

start_time = time.time()
model.fit(X, Y)
end_time = time.time()

Ypred = model.predict(X).astype(int)

print("Train Error: {:.5f}\n".format(error(Y, Ypred)))
print("Training time: %s seconds\n" % (end_time-start_time))

pred_y = model.predict(test_X).astype(int)

print("Test Error: {:.5f}\n".format(error(test_Y, pred_y)))

output = pd.DataFrame({'Actual': Y, 'Train Prediction': Ypred})
output2 = pd.DataFrame({'Test': np.array(test_Y), 'Test Prediction': pred_y})
output.to_csv('train_ou.csv', index=False)
output.to_csv('test_out.csv', index=False)

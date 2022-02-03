import numpy as np
import tensorflow as tf
from tensorflow import keras as K

import pandas as pd

from model import ProTran

import matplotlib.pyplot as plt

### Data 
df_stock = pd.read_csv("../stock_data/stock_sample.csv", index_col=0)

def stock_data_generator(df, k, tau):
    n = df.shape[0] - k - tau
        
    input_data = np.zeros((n, k, df.shape[1]), dtype=np.float32)
    target = np.zeros((n, 1, tau), dtype=np.float32)
    infer_data = np.zeros((n, k+tau, df.shape[1]), dtype=np.float32)

    for i in range(n):
        input_data[i, :, :] = df.iloc[i:i+k, :]
        target[i, :, :] = df.iloc[i+k:i+k+tau, 0]
        infer_data[i, :, :] = df.iloc[i:i+k+tau, :]   
    return input_data, target, infer_data

k = 10
tau = 10

input_stock, y_stock, infer_stock = stock_data_generator(df_stock, k, tau)


train_sample = input_stock[:120,:]
train_output = y_stock[:120, :]
train_infer = infer_stock[:120,:]

test_sample = input_stock[120:, :]
test_output = y_stock[120:, :]
test_infer = infer_stock[120:, :]

### Model
protran_model = ProTran(5, 60, 16, 8, 5, 4, 2)

def compute_loss(model, x):
  x_pred, _, gen_mean, gen_var, __, inf_mean, inf_var = model(x)
  kl = tf.reduce_mean(tf.math.log(gen_var/inf_var) + (inf_var + tf.math.square(inf_mean - gen_mean))/(2*gen_var))  
  recon = tf.reduce_mean(tf.math.square(x_pred[:, -1, :, :] - x))
  return kl + recon

train_loss = K.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

@tf.function
def train_step(model, x):
    with tf.GradientTape(persistent=True) as tape:
        loss = compute_loss(protran_model, train_infer)
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    train_loss(loss) 

### Train
EPOCHS = 3000

for epoch in range(EPOCHS):
    train_step(protran_model, train_infer)
    
    if (epoch + 1) % 50 == 0 : 
        template = 'EPOCH: {0}, Train Loss: {1:0.4f}'
        print(template.format(epoch+1, train_loss.result()))

x_pred, _, gen_mean, gen_var, __, inf_mean, inf_var = protran_model(train_infer)

plt.plot(np.arange(tf.reshape(x_pred[::8, :, :, 0], [-1]).shape[0]), tf.reshape(x_pred[::8, :, :, 0], [-1]).numpy(), color='orange', label='')
plt.plot(np.arange(tf.reshape(x_pred[::8, :, :, 0], [-1]).shape[0]), np.reshape(train_infer[::8, :, 0], [-1]), color='green', label='')
plt.legend()
plt.show()

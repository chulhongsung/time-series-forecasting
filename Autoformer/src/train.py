import tensorflow as tf
from tensorflow import keras as K
from model import Autoformer

autoformer = Autoformer(2, 2, 3, 20, 2, 20)

train_loss = K.metrics.Mean(name='train_loss')

mse = K.losses.MeanSquaredError()

@tf.function
def train_step(model, input, target):
    with tf.GradientTape(persistent=True) as tape:
        predicted = model(input)
        loss = mse(tf.squeeze(target), predicted)
    grad = tape.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(grad, model.weights))
    train_loss(loss) 

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

EPOCHS = 200

for epoch in range(EPOCHS):
    train_step(autoformer, train_sample, train_output)
    template = 'EPOCH: {}, Train Loss: {}'
    print(template.format(epoch+1, train_loss.result()))
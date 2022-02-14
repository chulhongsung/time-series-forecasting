import tensorflow as tf
from tensorflow import keras as K
from model import ProTran

protran_model = ProTran(d_embedding=5,
                        cat_dim=[12, 31],
                        d_model=24,
                        d_latent=16,
                        current_time=30,
                        num_heads=4,
                        num_layers=2)

def compute_loss(model, real_input, cate_input, beta):
    x_pred, _, gen_mean, gen_var, __, inf_mean, inf_var = model((real_input, cate_input))
    kl = tf.reduce_mean(tf.math.log(gen_var/inf_var) + (inf_var + tf.math.square(inf_mean - gen_mean))/(2*gen_var))  
    recon = tf.reduce_mean(tf.math.square(x_pred - real_input))
    return beta * kl + recon

train_loss = K.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)

@tf.function
def train_step(model, real_input, cate_input):
    with tf.GradientTape(persistent=True) as tape:
        loss = compute_loss(protran_model, real_input, cate_input, 1)
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    train_loss(loss) 
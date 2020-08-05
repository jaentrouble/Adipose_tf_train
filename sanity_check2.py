import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import adipose_models as models
from model_tools import load_dataset, AdiposeModel, load_valset
import model_lr
import datetime
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-mf', dest='mixed_f', action='store_true', default=False)
parser.add_argument('--name',dest='name', default=None)
parser.add_argument('--epochs',dest='epochs', default=10)
parser.add_argument('--steps',dest='steps', default=1000)
parser.add_argument('--model', dest='model')
parser.add_argument('-lr', dest='lr')
args = parser.parse_args()

MODEL = getattr(models, args.model)
LR = getattr(model_lr, args.lr)
if args.mixed_f:
    policy = mixed_precision.Policy('mixed_float16')
    print('policy: mixed_float16')
else:
    policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

st = time.time()

inputs = keras.Input((200,200,3))
mymodel = AdiposeModel(inputs, MODEL)
loss = keras.losses.BinaryCrossentropy(from_logits=True)
mymodel.compile(
        optimizer='adam',
        loss=loss,
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.1),
                keras.metrics.MeanSquaredError()],
    )
if args.name == None:
    log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
else :
    log_name = args.name
log_dir = "logs/fit/" + log_name

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      profile_batch='3,5',
                                                      update_freq='epoch')
lr_callback = keras.callbacks.LearningRateScheduler(LR, verbose=1)
lr_no_update = keras.callbacks.LearningRateScheduler(model_lr.lr_no_update, verbose=1)
train_ds = load_dataset('train_image', 1000)

Trash = train_ds.take(int(args.steps))
for img, msk in Trash.take(1).as_numpy_iterator():
    fig = plt.figure(figsize=(18,16))
    ax = fig.add_subplot(1,3,1)
    ax.imshow(img[0])
    ax = fig.add_subplot(1,3,2)
    ax.imshow(msk[0], cmap='binary')
    ax = fig.add_subplot(1,3,3)
    ax.imshow(mymodel.predict(img)[0], cmap='binary')
    plt.savefig('start')

mymodel.fit(
    x=Trash,
    epochs=int(args.epochs),
    # steps_per_epoch=int(args.steps),
    # validation_data=load_valset('val_image',int(args.steps)//10 +1),
    # validation_steps=int(args.steps),
    callbacks=[tensorboard_callback, lr_callback],
)

for img, msk in Trash.take(1).as_numpy_iterator():
    fig = plt.figure(figsize=(18,16))
    ax = fig.add_subplot(1,3,1)
    ax.imshow(img[0])
    ax = fig.add_subplot(1,3,2)
    ax.imshow(msk[0], cmap='binary')
    ax = fig.add_subplot(1,3,3)
    ax.imshow(mymodel.predict(img)[0], cmap='binary')
    plt.savefig('end')

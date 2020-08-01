import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sanity_models import *
from model_tools import load_dataset
import datetime
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

inputs = keras.Input((200,200,3))
mymodel = sanity_conv(inputs)
mymodel.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[keras.metrics.BinaryAccuracy()],
    )
mymodel.summary()
log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + log_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      profile_batch='3,5')

ds = load_dataset('train_image',1000)
Trash = ds.take(30)
mymodel.fit(
    x=Trash,

    epochs=1,
    # steps_per_epoch=int(args.steps),
    callbacks=[tensorboard_callback],
)
for img, msk in ds.take(2).as_numpy_iterator():
    fig = plt.figure(figsize=(18,16))
    ax = fig.add_subplot(1,3,1)
    ax.imshow(img[0])
    ax = fig.add_subplot(1,3,2)
    ax.imshow(msk[0], cmap='binary')
    ax = fig.add_subplot(1,3,3)
    ax.imshow(mymodel.predict(img), cmap='binary')
    plt.savefig('test')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sanity_models import *
from model_tools import load_dataset
import datetime
import time
import argparse
import numpy as np

inputs = keras.Input((200,200,3))
mymodel = sanity_conv(inputs)
mymodel.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[keras.metrics.BinaryAccuracy()],
    )
mymodel.summary()
log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      profile_batch='3,5')


Trash_X = np.random.randint(0,256,(1000,200,200,3), dtype=np.uint8)
Trash_Y = np.random.randint(0,2,(1000,100,100), dtype=np.float32)
mymodel.fit(
    x=Trash_X,
    y=Trash_Y,
    epochs=10,
    # steps_per_epoch=int(args.steps),
    callbacks=[tensorboard_callback],
)

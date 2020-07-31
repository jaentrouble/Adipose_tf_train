import tensorflow as tf
from tensorflow import keras
from adipose_models import *
from model_tools import load_dataset
import datetime
import time
st = time.time()

inputs = keras.Input((200,200,3))
mymodel = full_conv1(inputs)
mymodel.summary()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      profile_batch='5,7')
mymodel.fit(
    x=load_dataset('Adipose_train.tfrecords', 10000),
    epochs=20,
    steps_per_epoch=500,
    callbacks=[tensorboard_callback],
)
print('Took : {} seconds'.format(time.time()-st))

# tensorflow-cpu : 79.46 seconds/10step
# tensorflow-rocm : 36.88 seconds/10step

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from adipose_models import *
from model_tools import load_dataset
import datetime
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mf', dest='mixed_f', action='store_true', default=False)
parser.add_argument('--name',dest='name', default=None)
parser.add_argument('--epochs',dest='epochs', default=10)
parser.add_argument('--steps',dest='steps', default=1000)
args = parser.parse_args()

if args.mixed_f:
    policy = mixed_precision.Policy('mixed_float16')
    print('policy: mixed_float16')
else:
    policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

st = time.time()

inputs = keras.Input((200,200,3))
mymodel = full_conv1(inputs)
mymodel.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[keras.metrics.BinaryAccuracy()],
    )
mymodel.summary()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      profile_batch='3,5')
mymodel.fit(
    x=load_dataset('train_image', 10000),
    epochs=args.epochs,
    steps_per_epoch=args.steps,
    callbacks=[tensorboard_callback],
)
if args.name == None:
    mymodel.save('saved_model/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
else:
    mymodel.save('saved_model/' + args.name)

print('Took : {} seconds'.format(time.time()-st))

# tensorflow-cpu : 79.46 seconds/10step
# tensorflow-rocm : 36.88 seconds/10step

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import adipose_models as models
from model_tools import load_dataset, AdiposeModel, load_valset
from model_lr import *
import datetime
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mf', dest='mixed_f', action='store_true', default=False)
parser.add_argument('--name',dest='name', default=None)
parser.add_argument('--epochs',dest='epochs', default=10)
parser.add_argument('--steps',dest='steps', default=1000)
parser.add_argument('--model', dest='model')
args = parser.parse_args()

MODEL = getattr(models, args.model)
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
lr_callback = keras.callbacks.LearningRateScheduler(lr_step, verbose=1)

if args.name == None:
    mymodel.save('saved_model/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'/default')
else:
    mymodel.save('saved_model/' + args.name +'/default')


mymodel.fit(
    x=load_dataset('train_image', 3000),
    epochs=int(args.epochs),
    steps_per_epoch=int(args.steps),
    validation_data=load_valset('val_image',int(args.steps)//10 +1),
    validation_steps=int(args.steps),
    callbacks=[tensorboard_callback],
)
if args.name == None:
    mymodel.save('saved_model/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
else:
    mymodel.save('saved_model/' + args.name)

print('Took : {} seconds'.format(time.time()-st))

# tensorflow-cpu : 79.46 seconds/10step
# tensorflow-rocm : 36.88 seconds/10step

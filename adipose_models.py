import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as mylayers

# Get inputs and return outputs

def full_conv1(inputs):
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x, mask2 = mylayers.MaxPoolWithArgmax2D()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x, mask3 = mylayers.MaxPoolWithArgmax2D()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(256, 3, padding='same', activation='relu')(x)
    x = mylayers.Max_Unpool2D()(x, mask3)
    x = layers.Conv2DTranspose(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(x)
    x = mylayers.Max_Unpool2D()(x, mask2)
    x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def full_conv2(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    # DO NOT USE ReLU at the last layer!!!
    x = layers.Conv2D(1, 3, padding='same', activation='relu')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def full_conv3(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 3, padding='same', activation='relu')(x)
    # Relu was the problem!!!
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def full_conv3_fix(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def full_conv4(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def full_conv4_2(inputs):
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def full_conv5(inputs):
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

def full_conv5_2(inputs):
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32')(x)
    return outputs

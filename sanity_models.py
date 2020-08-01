import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as mylayers


def sanity_conv(inputs):
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 3, padding='same', activation='linear')(x)
    # x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('softmax', dtype='float32')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
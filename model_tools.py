import tensorflow as tf

IMG_SIZE = [200,200,3]
MASK_SIZE = [100,100]

def _parse_function(example_proto):
    feature_description = {
        'img' : tf.io.FixedLenSequenceFeature(IMG_SIZE, allow_missing=True,
                                        dtype=tf.int64, default_value=0),
        'mask' : tf.io.FixedLenSequenceFeature(MASK_SIZE, allow_missing=True,
                                        dtype=tf.int64, default_value=0),
    }
    parsed_feature = tf.io.parse_single_example(example_proto, feature_description)

    x = tf.cast(parsed_feature['img'], tf.float32) / 255.0
    x = tf.reshape(x, IMG_SIZE)
    y = tf.cast(parsed_feature['mask'], tf.float32)
    y = tf.reshape(y, MASK_SIZE)
    return x, y

def load_dataset(input_path, shuffle_buffer):
    autotune = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(input_path)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_parse_function, num_parallel_calls=autotune)
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()
    return dataset
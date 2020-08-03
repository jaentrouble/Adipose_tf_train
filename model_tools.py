import tensorflow as tf
import imageio as io
from os import listdir, path
import albumentations as A
import random
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = [200,200,3]
MASK_SIZE = [100,100]
WALL_COLOR = [0,0,0]
FILL_COLOR = [255,255,0]
OFFSET = 0

# def _parse_function(example_proto):
#     feature_description = {
#         'img' : tf.io.FixedLenSequenceFeature(IMG_SIZE, allow_missing=True,
#                                         dtype=tf.int64, default_value=0),
#         'mask' : tf.io.FixedLenSequenceFeature(MASK_SIZE, allow_missing=True,
#                                         dtype=tf.int64, default_value=0),
#     }
#     parsed_feature = tf.io.parse_single_example(example_proto, feature_description)

#     x = tf.cast(parsed_feature['img'], tf.float32) / 255.0
#     x = tf.reshape(x, IMG_SIZE)
#     y = tf.cast(parsed_feature['mask'], tf.float32)
#     y = tf.reshape(y, MASK_SIZE)
#     return x, y

class SampleGenerator():
    def __init__(self, image_dir):
        self.load_images(image_dir)
        self.aug = A.Compose([
            A.Resize(1100,1300),
            A.RandomCrop(IMG_SIZE[0], IMG_SIZE[1], p=1),
            A.OneOf([
                A.RandomGamma((40,200),p=1),
                A.RandomBrightness(p=1),
                A.RandomContrast(p=1),
                A.RGBShift(p=1),
            ], p=0.8),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
        ])
        self.resize_to_mask = A.Resize(MASK_SIZE[0],MASK_SIZE[1])

    def __iter__(self):
        return self

    def __call__(self, image_dir):
        return self

    def __next__(self):
        idx = random.randrange(0, self.n)
        x_st, x_ed, y_st, y_ed = self.borders[idx]
        distorted = self.aug(
            image = self.images[idx][2*x_st:2*x_ed,2*y_st:2*y_ed],
            mask = self.masks[idx][x_st:x_ed,y_st:y_ed]
        )
        image = distorted['image']
        # Dummy image to use only mask function
        mask = self.resize_to_mask(image=image, mask=distorted['mask'])['mask']
        return image, mask

    def load_images(self, image_dir):
        """
        Load image
        This assumes the image to be twice bigger than the mask
        """
        # search image names
        self.image_names = []
        self.mask_names = []
        image_dir = image_dir.decode()
        for name in listdir(path.join(image_dir, 'original')):
            self.image_names.append(name)
            self.mask_names.append(name+'_mask.png')

        # load raw images
        self.images = []
        self.masks_raw = []
        for im_n, ms_n in zip(self.image_names, self.mask_names):
            self.images.append(io.imread(path.join(image_dir, 'original', im_n)))
            self.masks_raw.append(io.imread(path.join(image_dir, 'mask', ms_n)))
        
        self.n = len(self.images)
        # check borders to cut
        self.borders = []
        self.masks = [] # Bool masks
        target_color = FILL_COLOR
        offset = OFFSET
        # [0_begin, 0_end, 1_begin, 1_end]
        for image, mask in zip(self.images, self.masks_raw):
            for r in range(mask.shape[0]):
                if np.any(np.all(mask[r]==target_color,axis=-1)):
                    row_begin = max(r - offset,0)
                    break
            for r in range(mask.shape[0]-1,-1,-1):
                if np.any(np.all(mask[r]==target_color,axis=-1)):
                    row_end = r + offset
                    break
            for c in range(mask.shape[1]):
                if np.any(np.all(mask[:,c]==target_color,axis=-1)):
                    col_begin = max(c - offset,0)
                    break
            for c in range(mask.shape[1]-1,-1,-1):
                if np.any(np.all(mask[:,c]==target_color,axis=-1)):
                    col_end = c + offset
                    break
            self.borders.append([row_begin, row_end, col_begin, col_end])
            self.masks.append(
                np.all(mask==WALL_COLOR,axis=-1).astype(np.float32))

class ValGenerator(SampleGenerator):
    def __init__(self, image_dir):
        super().__init__(image_dir)
        self.aug = A.Compose([
            A.Resize(1100,1300),
            A.RandomCrop(IMG_SIZE[0], IMG_SIZE[1], p=1),
        ])


def cast_function(image, mask):
    x = tf.cast(image, tf.float32) / 255.0
    y = tf.cast(mask, tf.float32)
    return x, y

def load_dataset(image_dir, shuffle_buffer):
    autotune = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_generator(
        SampleGenerator,
        (tf.uint8, tf.float32),
        (tf.TensorShape(IMG_SIZE), tf.TensorShape(MASK_SIZE)),
        args=[image_dir],
    )
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(cast_function, num_parallel_calls=autotune)
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()
    return dataset

def load_valset(image_dir, shuffle_buffer):
    autotune = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_generator(
        ValGenerator,
        (tf.uint8, tf.float32),
        (tf.TensorShape(IMG_SIZE), tf.TensorShape(MASK_SIZE)),
        args=[image_dir],
    )
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(cast_function, num_parallel_calls=autotune)
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()
    return dataset

if __name__ == '__main__':
    ds = load_dataset('train_image',1)
    for img, msk in ds.take(2).as_numpy_iterator():
        fig = plt.figure(figsize=(18,16))
        ax = fig.add_subplot(1,2,1)
        ax.imshow(img[0])
        ax = fig.add_subplot(1,2,2)
        ax.imshow(msk[0], cmap='binary')
        plt.savefig('test')
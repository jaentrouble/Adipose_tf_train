import imageio as io
from os import listdir, path
import matplotlib.pyplot as plt
import numpy as np
import random
import albumentations as A
from tqdm import tqdm, trange

image_names = []
mask_names = []
IMAGE_DIR = 'train_image'

for name in listdir(path.join(IMAGE_DIR, 'original')):
    image_names.append(name)
    mask_names.append(name+'_mask.png')

images = []
masks = []
for im_n, ms_n in zip(image_names, mask_names):
    images.append(io.imread(path.join(IMAGE_DIR, 'original', im_n)))
    masks.append(io.imread(path.join(IMAGE_DIR, 'mask', ms_n)))

cut_image = []
cut_mask = []
offset = 0
target_color = [255,255,0]

for image, mask in zip(images, masks):
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

    cut_image.append(image[2*row_begin:2*row_end,2*col_begin:2*col_end])
    cut_mask.append(mask[row_begin:row_end,col_begin:col_end])

cut_mask_bool = []
wall_color = [0,0,0]

for c_mask in cut_mask:
    booled = np.all(c_mask==wall_color,axis=-1)
    twiced = np.repeat(np.repeat(booled, 2, axis=0),2, axis=1)
    cut_mask_bool.append(twiced)

aug = A.Compose([
    A.RandomCrop(200, 200, p=1),
    A.OneOf([
        A.RandomGamma((40,200),p=1),
        A.RandomBrightness(p=1),
        A.RandomContrast(p=1),
        A.RGBShift(p=1),
    ], p=0.8),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1),
])

X_data = np.empty((6000*len(cut_image),200,200,3),dtype=np.uint8)
Y_data = np.empty((6000*len(cut_image),100,100),dtype=np.bool)

pbar = tqdm(total=len(cut_image))
idx = 0
for img, msk_b in zip(cut_image, cut_mask_bool):
    for _ in trange(6000, leave=False):
        auged = aug(image=img, mask=msk_b)
        X_data[idx] = auged['image']
        Y_data[idx] = auged['mask'][::2,::2]
        idx += 1
    pbar.update()
print('X_data shape:{}'.format(X_data.shape))
print('X_data type:{}'.format(X_data.dtype))
print('Y_data shape:{}'.format(Y_data.shape))
print('Y_data type:{}'.format(Y_data.dtype))

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.imshow(X_data[100])
ax = fig.add_subplot(1,2,2)
ax.imshow(Y_data[100])
plt.savefig('sample.png', dpi=300)
import os
import gc
import datetime
import numpy as np
import cv2

from os import listdir
from os.path import isfile, join

from copy import deepcopy

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback, EarlyStopping
from keras import backend as K
from keras.utils import Sequence

from libs.pconv_model import PConvUnet

## ============================================================================
##                           Data Generator
## ============================================================================
class AugmentingDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, mask_directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None,
                                                shuffle=False,*args, **kwargs)
        generator_mask = super().flow_from_directory(mask_directory,
                                                     class_mode=None,
                                                     shuffle=False,*args,
                                                     **kwargs)
        seed = None if 'seed' not in kwargs else kwargs['seed']

        while True:
            # Get augmentend image samples
            ori = next(generator)
            mask = next(generator_mask)

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 255

            masked = masked / 255.
            mask = mask / 255.
            ori = ori / 255.

            gc.collect()
            yield [masked, mask], ori

## ============================================================================
##                           Predict and save images
## ============================================================================
def plot_callback(model, n, path, masked, mask, ori):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""

    # Get samples & Display them
    pred_img = model.predict([masked, mask])

    for i in range(len(ori)):
        cv2.imwrite(path+'input_'+str(i+1+n)+'.png', masked[i,:,:,:]*255)
        cv2.imwrite(path+'output_'+str(i+1+n)+'.png', pred_img[i,:,:,:]*255)
        cv2.imwrite(path+'gt_'+str(i+1+n)+'.png', ori[i,:,:,:]*255)

## ============================================================================
##                        Command line arguments
## ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", "-g", type=int, default=4,
                                        help="Number of GPUs to use, default=4")
parser.add_argument("--root", "-r", type=str, default='./', help="root path")
args = parser.parse_args()

GPUS = args.gpu
BATCH_SIZE = GPUS
n_test = 239

steps_test = n_test//BATCH_SIZE

## ============================================================================
##                                  PATHS
## ============================================================================
PATH = args.root
save_path = PATH+'outputs/'
MASK_TEST_DIR = PATH+'dataset/mask/full/test/'
IM_TEST_DIR  = PATH+'dataset/gt/full/test/'

## ============================================================================
##                              Data generator
## ============================================================================
test_datagen = AugmentingDataGenerator()
test_generator = test_datagen.flow_from_directory(
                                                IM_TEST_DIR,
                                                MASK_TEST_DIR,
                                                target_size=(256, 256),
                                                color_mode = 'grayscale',
                                                batch_size=BATCH_SIZE,
                                                seed=42)
                                                
## ============================================================================
##                           Load model
## ============================================================================
model = PConvUnet(vgg_weights=None, inference_only=True)
model.load(PATH+'phase2_gray/weights.h5', train_bn=False)

## ============================================================================
##                           Predict
## ============================================================================
for n in range(steps_test):
    test_data = next(test_generator)
    (masked, mask), ori = test_data
    plot_callback(model, n, save_path, masked, mask, ori)

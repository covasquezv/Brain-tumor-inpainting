import os
import gc
import datetime
import numpy as np

from copy import deepcopy

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence

# Change to root path
if os.path.basename(os.getcwd()) != 'Keras':
    os.chdir('..')

from libs.pconv_model import PConvUnet
import libs.my_callback as my_callback

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
            masked[mask==0] = 255.
            masked = masked / 255.
            mask = mask / 255.
            ori = ori / 255.

            gc.collect()
            yield [masked, mask], ori

## ============================================================================
##                        Command line arguments
## ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-bs", type=int, default=4,
                                        help="Batch Size per GPU, default=4")
parser.add_argument("--gpu", "-g", type=int, default=4,
                                        help="Number of GPUs to use, default=4")
parser.add_argument("--root", "-r", type=str, default='./', help="root path")
args = parser.parse_args()

GPUS = args.gpu
batch_size = args.batch_size
PATH = args.root

## ============================================================================
##                              PATHS
## ============================================================================

DIR_MASK = PATH+'dataset/mask/'
DIR_IM = PATH+'dataset/gt/'

IM_TRAIN_DIR = DIR_IM+'train'
IM_VAL_DIR = DIR_IM+'val'
IM_TEST_DIR = DIR_IM+'test'

MASK_TRAIN_DIR = DIR_MASK+'train'
MASK_VAL_DIR = DIR_MASK+'val'
MASK_TEST_DIR = DIR_MASK+'test'

## ============================================================================
##                              Some settings ...
## ============================================================================

if GPUS > 0 : BATCH_SIZE = batch_size*GPUS
else: BATCH_SIZE = batch_size

## ============================================================================
##                              Data generators
## ============================================================================

# Create training generator
train_datagen = AugmentingDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
train_generator = train_datagen.flow_from_directory(
    IM_TRAIN_DIR,
    MASK_TRAIN_DIR,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    seed=42
)

# Create validation generator
val_datagen = AugmentingDataGenerator()
val_generator = val_datagen.flow_from_directory(
    IM_VAL_DIR,
    MASK_VAL_DIR,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    seed=42
)

## ============================================================================
##                              Training
## ============================================================================

## -------------  Phase 1 - with batch normalization - lr 2e-4 -----------------
print('Starting phase 1')
# output path
FOLDER = PATH+'phase1_GRAY/'

## ********************* Model definition **************************************
model = PConvUnet(vgg_weights=PATH+'vgg_grayscale.h5',
                  gpus=GPUS)

## ************************* Callbacks *****************************************
checkpoint = ModelCheckpoint(
                            FOLDER+'weights.h5',
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=10,
                               restore_best_weights=True)

history = my_callback.Histories()

callbacks = [checkpoint, early_stopping, history]

## *********************** Train settings **************************************
steps_train = 5000
steps_val = 1700
epochs = 1000

## ************************** Model fit ****************************************
model.fit_generator(
                    train_generator,
                    steps_per_epoch=steps_train,
                    validation_data=val_generator,
                    validation_steps=steps_val,
                    epochs=epochs,
                    verbose=2,
                    workers=16,
                    use_multiprocessing=True,
                    callbacks=callbacks)

## ************************* save curves ***************************************
history_dictionary_loss = history.loss
history_dictionary_val_loss = history.val_loss
np.save(FOLDER+'loss.npy', history_dictionary_loss)
np.save(FOLDER+'val_loss.npy', history_dictionary_val_loss)

print('Ending phase 1')
## -------------  End phase 1 - with batch normalization -----------------------

## ----------  Phase 2 - batch normalization off - lr 5e-5 ---------------------
print('Starting phase 2')
# output path
FOLDER = PATH+'phase2_GRAY/'

## ********************* Model definition **************************************
model = PConvUnet(vgg_weights=PATH+'vgg_grayscale.h5',
                  gpus=GPUS)
model.load(
            FOLDER+'weights.h5',
            train_bn=False,
            lr=0.00005)

## ************************* Callbacks *****************************************
checkpoint = ModelCheckpoint(
                            FOLDER+'weights.h5',
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=10,
                               restore_best_weights=True)

history = my_callback.Histories()

callbacks = [checkpoint, early_stopping, history]

## *********************** Train settings **************************************
steps_train = 5000
steps_val = 1700
epochs = 1000

## ************************** Model fit ****************************************
model.fit_generator(
                    train_generator,
                    steps_per_epoch=steps_train,
                    validation_data=val_generator,
                    validation_steps=steps_val,
                    epochs=epochs,
                    verbose=2,
                    workers=16,
                    use_multiprocessing=True,
                    callbacks=callbacks)

## ************************* save curves ***************************************
history_dictionary_loss = history.loss
history_dictionary_val_loss = history.val_loss
np.save(FOLDER+'loss.npy', history_dictionary_loss)
np.save(FOLDER+'val_loss.npy', history_dictionary_val_loss)

print('Ending phase 2')

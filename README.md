# Brain tumor inpainting using Partial Convolutions

Implementation of ["Image Inpainting for Irregular Holes Using Partial Convolutions"]( https://arxiv.org/abs/1804.07723) for non-tumor tissue inpainting using Keras.
The implementation of this work is mainly based on [Mathias Gruber implementation](https://github.com/MathiasGruber/PConv-Keras) for natural images. Modifications have been made to perform inpainting in grayscale medical images. The trained PConv-net is able to remove brain tumors when the irregular mask is placed over the tumor.

VGG-16 weights transformation was made based on the method implemented by Rohit Saha, available [here](https://github.com/RohitSaha/VGG_Imagenet_Weights_GrayScale_Images). Mean and variance for grayscale image preprocessing was calculated through the luminosity formula:

*gray_val = (value_in_red * 0.2989) + (value_in_green * 0.5870) + (value_in_blue * 0.1140)*

### Data

For this work a brain tumor dataset containing 3064 T1-weighted contrast-inhanced images
was used. The dataset contains 2D slices images of 3 kinds of tumor (meningioma, glioma and pituitary tumor) from 3 anatomical planes (axial, sagittal and coronal), with its respective tumor manual delineations.
The dataset it's publicly available [here](https://figshare.com/articles/brain_tumor_dataset/1512427). Further information about the images can be found in the paper "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition".

Additionally, in this work multiple irregular masks were generated to train the image inpainting model. If you need access to the masks, please personally contact me (covasquezv@inf.udec.cl).

### Pre-trained weights

Weights for non-tumor brain tissue inpainting are available at:
* [VGG-16 weights tranformed to grayscale](https://drive.google.com/file/d/1Xi-cKUia9PJeJTMf7oGM8DsyzTKnNKfU/view?usp=sharing)
* [Phase 1](https://drive.google.com/drive/folders/1cIbpfStEtIEPl4JemmXDjwFrdJL4_tDB?usp=sharing) - training (Learning rate 0.0002)
* [Phase 2](https://drive.google.com/drive/folders/1mGUMMuSmI3OMSupiWsm5he7r7eVN-561?usp=sharing) - training (Learning rate 0.00005, batch normalization disabled in encoder)

Run `python predict.py` to obtain the prediction. More information about script arguments running `python predict.py -h`

### Architecture and loss function

For further information about network and loss function detailes please go to the [original paper](https://arxiv.org/pdf/1804.07723.pdf).

### Train using your own data

You can train the model using your own data by placing the images and masks in its corresponding train, val and test folders in the `dataset` path.
* `dataset/gt` : ground truth images (in this case, MRI)
* `dataset/mask` : irregular binary masks (must have same filename than ground truth image)

To run the code:
```
python main.py 
```

Additionally, run ` python main.py -h ` to get more information about the arguments and its default values.







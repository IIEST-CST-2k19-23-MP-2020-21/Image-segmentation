# Image-segmentation
Python code for segmentation of satellite images

# First Step

Basic models - Image masking, U-net, CNNs

**U-Net Model**

The u-net model is a convolutional neural network that takes an input image and outputs an image of the same size. Imagine the image to be a 3d matrix specifying pixels values, having dimensions (height * width * depth).

<div align="center">

<img src="https://raw.githubusercontent.com/nirabhromakhal/media/main/.github/images/image.png">

</div>

**Fig**: *A sample U-Net architecture*

The architecture of this network is U-shaped. It is symmetric and consists of two major parts — the left part is called contracting path, which is constituted by the general convolutional process; the right part is expansive path, which is constituted by upsampling blocks.
In the first part, it applies several downscaling operations which decreases the height and width, but increases the depth(number of channels) of the image. The depth of the model is pretty large at this point and allows the model to learn the features of the images. Then we have several upsampling operations which decrease the depth and increase width and height of the image. After the final upsampling, the image output dimensions are the same as the input dimensions.


*For our U-net model code:*

We have used tensorflow.keras.layers

We have 5 downsampling blocks. The first block takes the input image. In each downsampling block, we have two 3x3 convolutions, each convolution followed by batch normalization and activated with ‘elu’. Finally we have a max pooling operation which reduces the height and width of the image for next block.

The third downsampling block is shown below:
```
conv3 = layer.Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(pool2)
conv3 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv3)
conv3 = layer.advanced_activations.ELU()(conv3)
conv3 = layer.Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv3)
conv3 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv3)
conv3 = layer.advanced_activations.ELU()(conv3)
pool3 = layer.MaxPooling2D(pool_size=(2, 2))(conv3)
```

Next we have 4 upsampling blocks. In each upsampling block, we combine the final layer of previous block with a final layer (the last layer before max pooling) of a downsampling block (which is at same level), and this produces a layer with increased height and width. Now we again have two 3x3 convolutions, each convolution followed by batch normalization and activated with ‘elu’, on this upscaled layer. For the last upsampling block, the final layer is subjected to a 1x1 convolution operation with ‘sigmoid’ activation. The ‘sigmoid’ activation produces all values in the range of [0,1].

The second upsampling block is shown below:
```
up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
conv7 = layer.Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(up7)
conv7 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv7)
conv7 = layer.advanced_activations.ELU()(conv7)
conv7 = layer.Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv7)
conv7 = layer.normalization.BatchNormalization(mode=0, axis=1)(conv7)
conv7 = layer.advanced_activations.ELU()(conv7)
```

Finally we create the model using the input layer (input image), and the final layer of the final upscaling block. The model can then be compiled and trained on our dataset.


# Second Step

Constructing segmentation models for single class - building

<div align="center">

<img src="https://user-images.githubusercontent.com/72441280/117532960-a25a0400-b007-11eb-9ad5-01365ee75be4.png">

<img src="https://user-images.githubusercontent.com/72441280/117533087-65dad800-b008-11eb-8f58-88c594348fd8.png">

</div>
 
<div align="center">

<img src="https://user-images.githubusercontent.com/72441280/117533221-46907a80-b009-11eb-98c6-e5425c5ed173.png">

<img src="https://user-images.githubusercontent.com/72441280/117533312-bbfc4b00-b009-11eb-9d53-a7e0177c42ce.png">

</div>

 
# Third Step

Final model

<div align="center">

<img src="https://user-images.githubusercontent.com/72441280/117533564-21047080-b00b-11eb-908e-6109c445e780.png">

<img src="https://user-images.githubusercontent.com/72441280/117533583-3d081200-b00b-11eb-897e-a4fee6bd4c5b.png">

<img src="https://user-images.githubusercontent.com/72441280/117533609-5f019480-b00b-11eb-9bc6-d18bfe0c6ebb.png">

<img src="https://user-images.githubusercontent.com/72441280/117533620-688afc80-b00b-11eb-84f7-8138ec06f430.png">

<img src="https://user-images.githubusercontent.com/72441280/117533868-97ee3900-b00c-11eb-9180-20c83a725b5f.png">

</div>




# Datasets used to train and test the models

 - https://www.crowdai.org/challenges/mapping-challenge
 - DSTL Satellite Image Dataset

# Image-segmentation
Python code for segmentation of satellite images

# First Step

Basic models - Image masking, U-net, CNNs

## U-Net Model

The u-net model is a convolutional neural network that takes an input image and outputs an image of the same size. Imagine the image to be a 3d matrix specifying pixels values, having dimensions (height * width * depth).

<div align="center">

<img src="https://raw.githubusercontent.com/nirabhromakhal/media/main/.github/images/image.png">

</div>

**Fig**: *A sample U-Net architecture*

The architecture of this network is U-shaped. It is symmetric and consists of two major parts — the left part is called contracting path, which is constituted by the general convolutional process; the right part is expansive path, which is constituted by upsampling blocks.
In the first part, it applies several downscaling operations which decreases the height and width, but increases the depth(number of channels) of the image. The depth of the model is pretty large at this point and allows the model to learn the features of the images. Then we have several upsampling operations which decrease the depth and increase width and height of the image. After the final upsampling, the image output dimensions are the same as the input dimensions.


**For our U-net model code:**

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

## CNN MODEL 

Artificial neural networks (ANNs) comprises node , input layer , many hidden layers and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold.

![weight](https://user-images.githubusercontent.com/64643228/117578123-68255b00-b10a-11eb-8282-34ce5b1f079e.png)

 If this output of a node is above the specified threshold value, that node is activated. Otherwise, no data is passed along to the next layer of through this node in network.
![threshold](https://user-images.githubusercontent.com/64643228/117578129-72475980-b10a-11eb-9bbb-c0cde07fa60f.png)

Problem with ANN is that it has to learn very high number of paramter which depends on input size , which may count to billion parameter per image in a layer.

**CNN (Convolutional Neural Network)** has this advantage over ANN that it does not need to learn so many parameters and number of parameters that needed to learn depends on filter(/Kernel) size which does not depends on Input.

CNN comprises of three kinds of layers:<br>
    1. The **Convolutional** layer is the core building block of a CNN, and it is where the majority of computation occurs. It requires a few components, which are input data, a filter, and a feature map.<br>
![CNN Demo](https://media3.giphy.com/media/i4NjAwytgIRDW/giphy.gif?cid=790b76111e7adba87f0b31e88a7be8766e9c57eca1eec896&rid=giphy.gif&ct=g)<br>
    2. **Pooling** layer aims to decrease number of parameters in input. Again, we use a Kernel to swipe over input but this time it does not learn any parameter instead it applies some aggregation function over the receptive field.<br>
    3. **Fully Connected** layer connects each node in the output layer directly to a node in the previous layer (like layers in ANN).
    While convolutional and pooling layers tend to use ReLu functions, FC layers usually leverage a softmax activation function to classify inputs appropriately, producing a probability from 0 to 1

##### Code Explaination :
We are using *Keras functional API* to write our model. You can read about it [here](https://keras.io/guides/functional_api/).
Making an input node of image size + (3,) i.e if image size is 32 X 32 then our input node will be of size 32 X 32 X 3 (for rgb channels). 
```
ip_s = keras.Input(shape=img_size + (3,))
```
Spatial convolution over samples
```
x = layer.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same")(ip_s)
```
32 filters , each of size 3 x 3 is used , while keeping padding as "same" . Since , our strides is (2,2) , it will decrease our input size.

Now we will normalize the input recieved from previous layer using batch normalization . This will help aginst overfitting of data as well as it will faster the training process
(Read about [Batch Normalization](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/))

If a node will remain active in model is depends on its summed weighted input .
The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.
(Read about [Activation function](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/))

```
x = layer.BatchNormalization()(x)
# Rectified linear unit is used as activation function
x = layer.Activation("relu")(x)
```
Instead of having normal convolution layers in our model, we are including layers having Depthwise Seperable Convolution and then pointwise convolution. This will helps us in reducing computational complexity as well as faster training.
**layer.SeparableConv2D** is doing Depthwise Seperable Convolution
[Read about Depthwise Seperable Convolution](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
```
prev_block_activation = x

for filters in arr:
    # arr contains different values of filters
    x = layer.Activation("relu")(x)
    # spatial convolution followed by pointwise convolution
    x = layer.SeparableConv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = layer.BatchNormalization()(x)

    x = layer.Activation("relu")(x)
    x = layer.SeparableConv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = layer.BatchNormalization()(x)
    
    # Using max pooling
    # on 2D spatial data
    x = layer.MaxPooling2D(pool_size=, strides=(2, 2), padding="same")(x)
    res = layer.Conv2D(filters, kernel_size=(1, 1), strides=(2, 2), padding="same")(prev_block_activation)
    x = layer.add([x, res]) 
    # Update value
    prev_block_activation = x  
```

After doing downsampling in our model, we will do Upsampling. We are using *Nearest Neighbour* and *Transposed Convolution* for Upsampling.
[Transposed Convolution](https://towardsdatascience.com/transposed-convolution-demystified-84ca81b4baba)

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

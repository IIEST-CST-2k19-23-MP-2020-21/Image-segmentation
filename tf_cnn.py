import tensorflow.keras.layers as layer


def model(img_size, N):
    # N => number of classes in our classification model
    # img_size => height and width
    # colored samples
    ip_s = keras.Input(shape=img_size + (3,))

    # Downsampling
    
    # spatial convolution over samples
    x = layer.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same")(ip_s)
    # Using a 3X3 kernel(Can use 1X1, 5X5, 7X7, 9X9...)
    # No of filters the first layer learns from is 2^5
    # Can use hyper parameter tuning for different powers of 2
    
    # Normalization
    # Using default values for different parameters 
    # Diffrent values or parameters can be used while hyper parameter tuning
    x = layer.BatchNormalization()(x)
    # Rectified linear unit is used as activation function
    x = layer.Activation("relu")(x)

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

    # Upsampling
    for filters in arr:
        # arr contains different values for filters
        x = layer.Activation("relu")(x)
        x = layer.Conv2DTranspose(filters, kernel_size=(3, 3), padding="same")(x)
        x = layer.BatchNormalization()(x)

        x = layer.Activation("relu")(x)
        x = layer.Conv2DTranspose(filters, kernel_size=(3, 3), padding="same")(x)
        x = layer.BatchNormalization()(x)

        x = layer.UpSampling2D(2)(x)

        res = layer.UpSampling2D(2)(prev_block_activation)
        res = layer.Conv2D(filters, kernel_size=(1, 1), padding="same")(res)
        x = layer.add([x, res])  
        prev_block_activation = x 
        
    op_s = layer.Conv2D(N, (3, 3), activation="softmax", padding="same")(x)

    # model
    model = keras.Model(ip_s, op_s)
    return model


# keras.backend.clear_session()
model = model(img_size, N)

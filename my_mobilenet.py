from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.callbacks import *


def conv_block(tensor, channels, strides, alpha=1.0):
    channels = int(channels * alpha)
    x = Conv2D(channels, kernel_size=(3, 3), strides=strides, padding='same', kernel_initializer = "he_normal")(tensor)   
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def dw_block(tensor, channels, strides, alpha=1.0):
    channels = int(channels * alpha)
    # Depthwise
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same', depthwise_initializer = "he_normal")(tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Pointwise
    x = Conv2D(channels, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer = "he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def myMobileNetV1(input_shape, num_classes, alpha=1.0, include_top=True, weights=None):
    x_input = Input(shape = input_shape)
    x = conv_block(x_input, 32, (2, 2))

    #dw layer parameter
    layers = [(64, (1, 1)),  (128, (2, 2)), (128, (1, 1)), (256, (2, 2)), (256, (1, 1)), (512, (2, 2)), \
                *[(512, (1, 1)) for i in range(5)], (1024, (2, 2)),   (1024, (1, 1))]

    for i, (channels, strides) in enumerate(layers):
        x = dw_block(x, channels, strides)

    #add top layers    
    if include_top:
        x = GlobalAvgPool2D(name='global_avg')(x)
        x = Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=x_input, outputs=x)
    if weights is not None:
        model.load_weights(weights, by_name=True)
    return model
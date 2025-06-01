import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, initializers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Rescaling, MaxPooling2D, LeakyReLU
from tensorflow.keras import optimizers

def build_model(args, input_shape, n_categories):
    """
    Construct and compile the CNN given args, input shape, and number of classes.
    """

    # weight initializer map
    weightInitializers = { "he_normal" : initializers.HeNormal(), 
                      "normal" : initializers.RandomNormal(),
                      "xaviar": initializers.GlorotNormal()}
    kernelInitializer = weightInitializers[args.kernel_initializer.lower()]
    
    # build model
    model = Sequential([
        Input(shape=input_shape),
        Rescaling(1./255),

        # block 1
        Conv2D(args.n_convolution1, args.kernel_size1, strides=args.strides1,
               kernel_initializer=kernelInitializer,
               bias_initializer=initializers.Constant(args.bias_initializer), 
               name='conv1'),
        LeakyReLU(args.reluLeak1),
        BatchNormalization(),
        MaxPooling2D(pool_size=args.pool1),
        Dropout(args.dropout1),

        # block 2 (if present)
        *( [
            Conv2D(args.n_convolution2, args.kernel_size2, strides=args.strides2,
                   kernel_initializer=kernelInitializer,
                   bias_initializer=initializers.Constant(args.bias_initializer), 
                   name='conv2'),
            LeakyReLU(args.reluLeak2),
            BatchNormalization(),
            MaxPooling2D(pool_size=args.pool2),
            Dropout(args.dropout2)
        ] if args.n_convolution2>0 else [] ),

        # block 3 (if present)
        *( [
            Conv2D(args.n_convolution3, args.kernel_size3, strides=args.strides3,
                   kernel_initializer=kernelInitializer,
                   bias_initializer=initializers.Constant(args.bias_initializer), 
                   name='conv3'),
            LeakyReLU(args.reluLeak3),
            BatchNormalization(),
            MaxPooling2D(pool_size=args.pool3),
            Dropout(args.dropout3)
        ] if args.n_convolution3>0 else [] ),

        Flatten(),
        Dense(args.n_dense1,
              kernel_initializer=args.kernel_initializer,
              bias_initializer=initializers.Constant(args.bias_initializer)
              ),
        LeakyReLU(args.reluLeakDense1),
        BatchNormalization(),
        Dropout(args.dropoutDense1),

        Dense(n_categories, activation='softmax',
              kernel_initializer=args.kernel_initializer,
              bias_initializer=initializers.Constant(args.bias_initializer)
              )
    ])

    model.compile(
        optimizer=optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )
    print(model.summary())
    return model
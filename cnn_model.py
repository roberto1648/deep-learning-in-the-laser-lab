from keras.layers import Convolution1D, Dense, MaxPool1D, Dropout, Flatten
from keras.models import Input, Model


def model1(input_shape=(None, None, 2),
           conv_blocks=[{'nlayers': 2, 'nfilters': 8, 'kernel_size': 3},
                        {'nlayers': 2, 'nfilters': 16, 'kernel_size': 3},
                        {'nlayers': 3, 'nfilters': 32, 'kernel_size': 3}],
           dense_layers=[64, 8],
           nlabels=1,
           verbose=True):
    inp = x = Input(batch_shape=input_shape, name='input')

    for block_number, conv_block in enumerate(conv_blocks):
        for layer_number in range(conv_block['nlayers']):
            name = "conv_block_{}_layer_{}".format(block_number, layer_number)
            x = Convolution1D(conv_block['nfilters'], conv_block['kernel_size'],
                              name = name,
                              strides=1, activation='relu', padding='same')(x)
        x = MaxPool1D(2, name="max_pooling_{}".format(block_number))(x)

    x = Dropout(0.25, name="dropout_1")(x)
    x = Flatten(name = "flatten")(x)

    for layer_number, n_neurons in enumerate(dense_layers):
        name = "fc_{}".format(layer_number)
        x = Dense(n_neurons, activation='relu', name = name)(x)

    x = Dense(nlabels, activation='linear', name='predictions')(x)
    model = Model(inputs=inp, outputs=x)

    if verbose: print model.summary()

    return model


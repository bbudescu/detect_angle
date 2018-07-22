from keras.layers import Conv2D, BatchNormalization, Activation, SpatialDropout2D, MaxPooling2D, add
from keras.regularizers import l2
from keras import backend as K


def _block_name_base(stage, block):
    """Get the convolution name base and batch normalization name base defined by stage and block.

    If there are less than 26 blocks they will be labeled 'a', 'b', 'c' to match the paper and keras
    and beyond 26 blocks they will simply be numbered.
    """
    if block < 27:
        block = '%c' % (block + 97)  # 97 is the ascii number for lowercase 'a'
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    return conv_name_base, bn_name_base


def resnet_basic_block(filters, stage, block, dropout=None, is_first_block_of_first_layer=False):
    # resnet block, implemented as in keras_contrib
    conv_name_base, bn_name_base = _block_name_base(stage, block)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

    def f(input_features):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv2D(filters=filters, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4), name=conv_name_base + '2a')(input_features)
        else:
            x = BatchNormalization(axis=channel_axis, name=bn_name_base + '2a')(input_features)
            x = Activation("relu")(x)
            x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4), name=conv_name_base + '2a')(x)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = BatchNormalization(axis=channel_axis, name=bn_name_base + '2b')(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4), name=conv_name_base + '2b')(x)

        if filters == K.int_shape(input_features)[channel_axis]:
            shortcut = input_features
        else:
            shortcut = Conv2D(filters, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
                              name=conv_name_base + '1')(input_features)
            shortcut = BatchNormalization(axis=channel_axis, name=bn_name_base + '1')(shortcut)

        return shortcut + x

    return f


def modified_resnet_block(filters, base_name, block_idx, dropout):
    # replaces the subsampled convolutions with max pooling => fewer params => faster training
    # skip layer connections help avoiding vanishing gradients, although not as much as regular resnets. E.g., if the
    # net's output shape is (1, 1), then only a single pixel in each pool's gradient map will be updated by something
    # non-zero through the skip layer connections, if max pooling is used. Strangely enough, the same thing happens when
    # using strided convolutions with kernel of size (1,1) in the shortcuts of the classical resnet, though... However,
    # this doesn't happen for convolutions with kernel of size (3, 3) or (7, 7), despite the (2, 2) stride.

    # we obtain nets like this:
    #       +---+                 +---+
    #  +----| + |----+       +----| + |----+
    #  |    +---+    |       |    +---+    |
    #  |             v       |             v
    # -+->conv->conv-+->pool-+->conv->conv-+->pool->...
    conv_name_base, bn_name_base = _block_name_base(base_name, block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

    def f(input_features):
        x = BatchNormalization(axis=channel_axis, name=bn_name_base + '2a')(input_features)
        x = Activation("relu")(x)
        if dropout is not None:
            x = SpatialDropout2D(dropout)(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4), name=conv_name_base + '2a')(x)

        x = BatchNormalization(axis=channel_axis, name=bn_name_base + '2b')(x)
        x = Activation("relu")(x)
        if dropout is not None:
            x = SpatialDropout2D(dropout)(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4), name=conv_name_base + '2b')(x)

        if filters == K.int_shape(input_features)[channel_axis]:
            shortcut = input_features
        else:
            shortcut = Conv2D(filters, (1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
                              name=conv_name_base + '1')(input_features)
            shortcut = BatchNormalization(axis=channel_axis, name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])

        # previously we did this here instead of in the beginning
        # x = BatchNormalization(axis=channel_axis, name=bn_name_base + 'merged')(input_features)
        # x = Activation("relu")(x)

        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        return x

    return f






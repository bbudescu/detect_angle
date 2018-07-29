import math
from enum import Enum, auto, unique

from keras import backend as K
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation, Concatenate, Softmax
from keras.layers import Lambda
from keras.regularizers import l2
from keras.models import Model
from estimate_rotation.util import preprocess_input as preproc_custom
from keras.applications.inception_v3 import preprocess_input as preproc_inception
from keras.applications.vgg16 import preprocess_input as preproc_vgg
from keras.applications.resnet50 import preprocess_input as preproc_resnet


# from keras_applications.imagenet_utils import _obtain_input_shape

from estimate_rotation.common import AngleEncoding


@unique
class Bounding(Enum):
    # we can force net's xy representation (either output or penultimate layer) to [-1, 1] to match the range of target
    # sines and cosines
    NONE = auto()
    CLIP = auto()
    TANH = auto()
    NORM = auto()
    # we could also add 2 * sigmoid - 1, but it's almost the same as tanh


@unique
class Features(Enum):
    TRAIN = auto()  # build a model from scratch
    VGG16 = auto()
    RESNET50 = auto()
    INCEPTIONV3 = auto()


preproc = {
    Features.TRAIN: preproc_custom,
    Features.VGG16: preproc_vgg,
    Features.RESNET50: preproc_resnet,
    Features.INCEPTIONV3: preproc_inception
}


# we could bag all the feature maps at the end of the model if we wanted to handle images of different sizes without
# resizing them first, but this would complicate training, too, as we would have uneven batches. @TODO: try later
# @unique
# class Bagging(Enum):
#     NONE = auto()
#     AVG = auto()
#     MAX = auto()


def UntrainedFilterModel(img_side, grayscale, dropout=.25):
    from estimate_rotation.resnet import modified_resnet_block

    filters = 64  # like resnet50, vgg16, vgg19

    filter_increase_factor = 1.5  # with 2 we run out of memory, even for batches of a single image of 316 x 316

    # the first layer also subsamples by a factor of 4
    n_outs_layer1 = filters * img_side * img_side / 4.

    # we want to add layers until we reach under 50k outputs, to which we can add fc layers
    max_conv_outs = 20000

    scale = float(n_outs_layer1) / max_conv_outs

    # each block halves the image side, and doubles the number of feature maps, thus halving the total number of units
    block_scale = 4. / filter_increase_factor

    n_blocks = int(math.ceil(math.log(scale, block_scale)))

    # input_shape = [None] * 3
    input_shape = [img_side] * 3  # just for debug
    input_shape[0 if K.image_data_format() == 'channels_first' else -1] = 1 if grayscale else 3

    input_img = Input(input_shape, name='input')

    x = Conv2D(filters, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4), name='conv1')(input_img)

    for block_id in range(n_blocks):
        filters = int(filters * filter_increase_factor)
        x = modified_resnet_block(filters, '', block_id, dropout)(x)

    x = BatchNormalization(axis=1 if K.image_data_format() == 'channels_first' else 3)(x)
    x = Activation('relu')(x)

    feature_extractor = Model(input_img, x)
    frozen_layers = []

    return feature_extractor, frozen_layers


def PretrainedFilterModel(model):
    assert model in Features
    assert model != Features.TRAIN
    if model == Features.VGG16:
        from keras.applications.vgg16 import VGG16 as Model
    elif model == Features.RESNET50:
        from keras.applications.resnet50 import ResNet50 as Model
    elif model == Features.INCEPTIONV3:
        from keras.applications.inception_v3 import InceptionV3 as Model
    else:
        raise NotImplementedError()

    feature_extractor = Model(weights='imagenet', include_top=False)

    frozen_layers = []

    for layer in feature_extractor.layers:
        if layer.trainable:
            frozen_layers.append(layer)
            layer.trainable = False

    # actually, model summary just skips relu activations
    # if model == Features.VGG16:
    #     channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    #     x = BatchNormalization(channel_axis)(model.output)
    #     x = Activation('relu')(x)

    return feature_extractor, frozen_layers


def EncoderModel(input_shape, dropout=.25):
    feature_maps = Input(shape=input_shape)

    x = Flatten()(feature_maps)
    if dropout:
        x = Dropout(dropout)(x)
    x = Dense(4096, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = Dense(2048, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    model = Model(feature_maps, x)

    return model


def estimate_angle(angle_encoding, force_xy=None, bounding=None, n_classes=None, dropout=.25):
    assert angle_encoding in AngleEncoding

    if angle_encoding in {AngleEncoding.CLASSES, AngleEncoding.SINCOS}:
        assert force_xy is None
    else:
        assert force_xy is not None

    if angle_encoding == AngleEncoding.CLASSES:
        assert n_classes is not None
    else:
        assert n_classes is None

    if angle_encoding == AngleEncoding.CLASSES:
        assert bounding is None
    else:
        assert bounding in Bounding
        if not force_xy and angle_encoding != AngleEncoding.SINCOS:
            assert bounding != Bounding.NORM

    out_scale = None
    if force_xy:
        if K.backend() == 'theano':
            from theano.tensor import arctan2 as atan2
        elif K.backend() == 'tensorflow':
            from tensorflow import atan2
        else:
            raise NotImplementedError('backend ' + K.backend() + ' not supported')

        # output is always in -pi...pi (atan is applied in previous layer)
        if angle_encoding == AngleEncoding.UNIT:
            out_scale = 1 / math.pi
        elif angle_encoding == AngleEncoding.DEGREES:
            out_scale = 180 / math.pi
    elif bounding != Bounding.NONE:
        # force_xy is false, so x contains the outputs, which have been trimmed to [-1, 1]
        if angle_encoding == AngleEncoding.RADIANS:
            out_scale = math.pi
        elif angle_encoding == AngleEncoding.DEGREES:
            out_scale = 180

    def f(img_encoded, rot_encoded):
        x = Concatenate()([img_encoded, rot_encoded])  # 2048 + 2048 = 4096 inputs
        if dropout:
            x = Dropout(dropout)(x)

        x = Dense(2048, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if angle_encoding == AngleEncoding.CLASSES:
            n_units = n_classes
        elif angle_encoding == AngleEncoding.SINCOS or force_xy:
            n_units = 2
        else:
            n_units = 1  # degrees, radians or [-1, 1]

        if dropout:
            x = Dropout(dropout)(x)

        # we use he init even though the output is not relu. But input is, so it should be beter than xavier.
        x = Dense(n_units, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)

        if angle_encoding == AngleEncoding.CLASSES:
            return Softmax()(x)

        if bounding == Bounding.CLIP:
            x = Lambda(lambda inp: K.clip(inp, -1, 1), name='clip')(x)
        elif bounding == Bounding.TANH:
            x = Activation('tanh', name='clip')(x)
        elif bounding == Bounding.NORM:
            x = Lambda(lambda inp: inp / K.sqrt(K.sum(inp * inp, axis=-1)), name='clip')(x)

        if angle_encoding == AngleEncoding.SINCOS:
            return x

        if force_xy:
            x = Lambda(lambda xy: atan2(xy[:, 1], xy[:, 0]), name='atan2')(x)

        if out_scale:
            x = Lambda(lambda inp: inp * out_scale)(x)

        return x

    return f


def to_deg(angles, encoding, n_classes=None):
    from estimate_rotation.common import AngleEncoding
    # simplified version of function to_deg from model.py
    assert encoding in AngleEncoding
    if encoding == AngleEncoding.CLASSES:
        if not n_classes:
            n_classes = K.int_shape(angles)[-1]

        resolution = 360. / n_classes

    if encoding == AngleEncoding.SINCOS:
        if K.backend() == 'theano':
            from theano.tensor import arctan2 as atan2
        elif K.backend() == 'tensorflow':
            from tensorflow import atan2
        else:
            raise NotImplementedError('backend ' + K.backend() + ' not supported')

    if encoding == AngleEncoding.DEGREES:
        deg = angles
    elif encoding == AngleEncoding.RADIANS:
        deg = angles * 180 / math.pi
    elif encoding == AngleEncoding.UNIT:
        deg = angles * 180
    elif encoding == AngleEncoding.SINCOS:
        sines = angles[:, 0]
        cosines = angles[:, 1]
        rad = atan2(cosines, sines)
        deg = rad * 180 / math.pi
    elif encoding == AngleEncoding.CLASSES:
        deg = K.cast(K.argmax(angles, axis=-1), K.floatx()) * resolution + resolution / 2. - 180.
    else:
        raise NotImplementedError('unsupported angle encoding ' + str(encoding))

    return deg


def model(features, img_side, grayscale, angle_encoding=AngleEncoding.SINCOS, force_xy=None, bounding=Bounding.NONE,
          n_classes=None, dropout=None, decode_angle=True):
    assert features in Features
    if features != Features.TRAIN:
        # all the pretrained models use rgb
        assert not grayscale

    n_channels = 1 if grayscale else 3

    if K.image_data_format() == 'channels_first':
        input_shape = (n_channels, img_side, img_side)
    else:
        input_shape = (img_side, img_side, n_channels)

    imgs = Input(input_shape, name='input_img')
    rots = Input(input_shape, name='input_rot')

    if features == Features.TRAIN:
        feature_extractor, frozen_layers = UntrainedFilterModel(img_side, grayscale, dropout)
    else:
        feature_extractor, frozen_layers = PretrainedFilterModel(features)

    img_features = feature_extractor(imgs)
    rot_features = feature_extractor(rots)

    encoder = EncoderModel(K.int_shape(img_features)[1:], dropout)

    encoded_imgs = encoder(img_features)
    encoded_rots = encoder(rot_features)

    out = estimate_angle(angle_encoding, force_xy, bounding, n_classes, dropout)(encoded_imgs, encoded_rots)

    if decode_angle:
        out = Lambda(lambda encoded_angles: to_deg(encoded_angles, angle_encoding))(out)

    return Model([imgs, rots], out), frozen_layers


def tst_instantiation():
    # test all valid parameter combinations; this should also provide a map for sampling during hyperparameter
    # optimization
    resolution_degrees = .5
    min_img_side = int(math.ceil(158. / resolution_degrees))
    img_side = int(math.ceil(1.2 * min_img_side))

    for dropout in None, .25:
        for decode_angle in True, False:
            for grayscale in True, False:
                if grayscale:
                    allowed_features = [Features.TRAIN]
                else:
                    allowed_features = Features

                for features in allowed_features:
                    for angle_encoding in AngleEncoding:
                        if angle_encoding == AngleEncoding.CLASSES:
                            n_classes = 360. / resolution_degrees
                            allowed_force_xy = [None]
                        else:
                            n_classes = None

                            if angle_encoding == AngleEncoding.SINCOS:
                                allowed_force_xy = [None]
                            else:
                                # angle_encoding is in {degrees, radians, unit}
                                allowed_force_xy = [True, False]

                        for force_xy in allowed_force_xy:
                            if force_xy is None:
                                if angle_encoding == AngleEncoding.CLASSES:
                                    allowed_bounding = [None]
                                else:
                                    # angle_encoding is SINCOS
                                    allowed_bounding = Bounding
                            else:
                                # angle_encoding is in {degrees, radians, unit}
                                if force_xy:
                                    allowed_bounding = Bounding
                                else:
                                    allowed_bounding = [Bounding.NONE, Bounding.CLIP, Bounding.TANH]

                            for bounding in allowed_bounding:
                                angle_estimator = model(features, img_side, grayscale, angle_encoding, force_xy,
                                                        bounding, n_classes, dropout, decode_angle)


def main():
    # user params

    resolution_degrees = .5

    # ~user_params

    # In order to determine the minimum size of the image that is fed to the neural net, we consider the resolution
    # required by an approach based on image registration. A keypoint detector without subpixel accuracy would require a
    # displacement of at least one pixel in order to disambiguate between the location of the same keypoint before and
    # after rotation. We, thus, formulate the constraint that at least half of the pixels that are contained by both
    # images to have the minimally required displacement when rotating by an angle equal to the desired resolution.
    # The largest margins after rotation (and, thus, the minimum common area) are obtained when rotating by 45 degrees.
    # Then we constrain the length, in pixels, of the circle centered in the image center that covers half of the
    # number of pixels in the minimum common area to be at least the number of resolution_degree intervals that fit in
    # the 360 degree possible rotation range.
    min_img_side = int(math.ceil(158. / resolution_degrees))

    # metaparams regarding net design

    # choose a slightly larger image size, as many pixels don't have fully populated neighbourhoods
    img_side = int(math.ceil(1.2 * min_img_side))

    angle_encoding = AngleEncoding.SINCOS

    # if we use no angle encoding, we can make the second last layer to be xy (~sincos) and then apply atan2 to get the
    # output angle, which we directly compare with the target; atan's derivative is 1 / (1 + x^2), which has a value of
    # 1 at 0, and falls to 0 rather quickly. This should prevent exploding gradients, but might slow down learning in
    # the beginning, when many angle differences might be very high (above pi). Only potential problem would be that
    # x and y are not implicitly bounded (atan2(y,x) ~ atan(y/x)) and might vanish / explode, but that might be fixed
    # by clipping them explicitly.
    force_xy = None  # False, True

    # whether or not to use global pooling (average or max) for bagging
    # bagging = Bagging.AVG

    grayscale = True

    bounding = Bounding.NONE

    features = Features.TRAIN

    # ~metaparams

    # metaparam check

    if angle_encoding == AngleEncoding.CLASSES:
        n_classes = int(math.ceil(360. / resolution_degrees))
    else:
        n_classes = None

    dropout = None

    # my best guesses, in order of preference (see below for a few tests):
    #
    # # in my opinion, the most stable encoding is sin cos. The target values are bounded, so they will constrain the
    # # output values, too
    # angle_estimator, _ = model(Features.TRAIN, img_side, grayscale=True, angle_encoding=AngleEncoding.SINCOS,
    #                         force_xy=None, bounding=Bounding.NONE, n_classes=None, dropout=None, decode_angle=True)
    #
    # del angle_estimator
    #
    # # slightly more risky, but with great potential: constrain the penultimate layer to xy and apply atan2. The risk
    # # comes from the fact that x and y values are not constrained to an interval by the output values, as atan is
    # # applied on y/x
    # angle_estimator, _ = model(Features.TRAIN, img_side, grayscale=True, angle_encoding=AngleEncoding.UNIT,
    #                         force_xy=True, bounding=Bounding.NONE, n_classes=None, dropout=None, decode_angle=True)
    #
    # del angle_estimator
    #
    # # if previous model fails because xy values and gradients blow up, try binding their values (and gradients)
    # # artificially
    # angle_estimator, _ = model(Features.TRAIN, img_side, grayscale=True, angle_encoding=AngleEncoding.UNIT,
    #                         force_xy=True, bounding=Bounding.TANH, n_classes=None, dropout=None, decode_angle=True)
    #
    # del angle_estimator

    # # after trying all these regressors, it's worth seeing how well a classifier works. Set n_classes to
    # # 360. / (avg_angle_error / 2)
    # # where avg_angle_error is the best error we got via regression
    # angle_estimator, _ = model(Features.TRAIN, img_side, grayscale=True, angle_encoding=AngleEncoding.CLASSES,
    #                         force_xy=None, bounding=None, n_classes=int(math.ceil(360. / 0.36)),
    #                         dropout=None, decode_angle=True)
    #
    # del angle_estimator

    # a few tests:

    angle_estimator, _ = model(features, img_side, grayscale, angle_encoding, force_xy, bounding, n_classes, dropout,
                               decode_angle=True)

    del angle_estimator
    #
    # grayscale = False
    #
    # angle_estimator, _ = model(features, img_side, grayscale, angle_encoding, force_xy, bounding, n_classes, dropout,
    #                         decode_angle=True)
    #
    # del angle_estimator
    #
    # angle_estimator, _ = model(Features.VGG16, img_side, grayscale, angle_encoding, force_xy, bounding, n_classes,
    #                         dropout, decode_angle=True)
    #
    # del angle_estimator
    #
    # angle_estimator, _ = model(Features.RESNET50, img_side, grayscale, angle_encoding, force_xy, bounding, n_classes,
    #                         dropout, decode_angle=True)
    #
    # del angle_estimator
    #
    # angle_estimator, _ = model(Features.INCEPTIONV3, img_side, grayscale, angle_encoding, force_xy, bounding, n_classes,
    #                         dropout, decode_angle=True)
    #
    # del angle_estimator


if __name__ == '__main__':
    main()

import math
import pickle

from keras.optimizers import nadam, sgd
from keras.losses import mse, categorical_crossentropy
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import load_model

from estimate_rotation.common import AngleEncoding
from estimate_rotation.model import Features, Bounding


def decode(angles, encoding, n_classes):
    from estimate_rotation.common import AngleEncoding
    # simplified version of function to_deg from model.py
    assert encoding in AngleEncoding
    if encoding == AngleEncoding.CLASSES:
        assert n_classes is not None
        resolution = 360. / n_classes
    else:
        assert n_classes is None

    if encoding == AngleEncoding.SINCOS:
        if K.backend() == 'theano':
            from theano.tensor import arctan2 as atan2
        elif K.backend() == 'tensorflow':
            from tensorflow import atan2
        else:
            raise NotImplementedError('backend ' + K.backend() + ' not supported')

    if encoding == AngleEncoding.DEGREES:
        return angles
    if encoding == AngleEncoding.RADIANS:
        return angles * (180 / math.pi)
    if encoding == AngleEncoding.UNIT:
        return angles * 180
    if encoding == AngleEncoding.SINCOS:
        return atan2(angles[:, 1], angles[:, 0]) * 180 / math.pi
    if encoding == AngleEncoding.CLASSES:
        return K.cast(K.argmax(angles, axis=-1), K.floatx()) * resolution + resolution / 2. - 180.
    else:
        raise NotImplementedError('unsupported angle encoding ' + str(encoding))


def abs_deg_diff(angles_pred, angles_true, encoding, n_classes):
    angles_pred = decode(angles_pred, encoding, n_classes)
    angles_true = decode(angles_true, encoding, n_classes)
    # angles are between -180 and 180 degrees
    # diffs are between -360 and 360 degrees
    # abs_diffs are between 0 and 360 degrees
    abs_diffs = K.abs(angles_true - angles_pred)
    return K.minimum(abs_diffs, 360 - abs_diffs)


def mean_abs_deg_diff(angles_pred, angles_true):
    return K.mean(abs_deg_diff(angles_pred, angles_true,
                               mean_abs_deg_diff.angle_encoding, mean_abs_deg_diff.n_classes))


def do_training(
        # training data params (in keras batch_size is passed to the dataset, although it's, technically, a training metaparameter)
        train_set, val_set, test_set, data_inmem, batch_size,
        # model params
        model, frozen_layers, angle_encoding, n_classes,
        # training metaparameters
        optimizer, lr, optimizer_kw_args, max_epochs,
        # output
        net_filename
):
    from estimate_rotation.common import AngleEncoding

    optimizer = optimizer(lr, **optimizer_kw_args)

    loss = categorical_crossentropy if angle_encoding == AngleEncoding.CLASSES else mse

    mean_abs_deg_diff.angle_encoding = angle_encoding
    mean_abs_deg_diff.n_classes = n_classes

    model.compile(optimizer=optimizer, loss=loss, metrics=[mean_abs_deg_diff])

    monitor = 'val_' + mean_abs_deg_diff.__name__
    epsilon = .005

    model_checkpoint = ModelCheckpoint(net_filename, monitor, verbose=1, save_best_only=True)

    # @TODO: for a real life scenario: create our own EarlyStopping implementation that supports decisions based on
    #        relative improvement, and that increases patience proportionally with the number of epochs so far; also,
    #        a minimum number of epochs should be allowed as param
    early_stoppping = EarlyStopping(monitor, min_delta=epsilon, patience=3, verbose=1)

    lr_schedule = ReduceLROnPlateau(monitor, factor=0.25, patience=1, verbose=1, min_delta=epsilon, cooldown=0,
                                    min_lr=1e-12)

    callbacks = [model_checkpoint, early_stoppping, lr_schedule]

    if data_inmem:
        [train_img, train_rot], train_angles = train_set
        model.fit([train_img, train_rot], train_angles, batch_size, max_epochs, callbacks=callbacks,
                  validation_data=val_set, shuffle=True)
    else:
        batches_per_epoch = (len(train_set.filenames) + batch_size - 1) // batch_size
        batches_per_val = (len(val_set.filenames) + batch_size - 1) // batch_size
        model.fit_generator(train_set, batches_per_epoch, max_epochs, callbacks=callbacks, validation_data=val_set,
                            validation_steps=batches_per_val)

    # load net with best xval
    model = load_model(net_filename, custom_objects={'mean_abs_deg_diff': mean_abs_deg_diff})

    for frozen_layer in frozen_layers:
        frozen_layer.trainable = True

    if frozen_layers:
        # TODO: continue training after unfreezing layers
        pass

    if model:
        [test_img, test_rot], test_angles = test_set
        test_metrics = model.evaluate([test_img, test_rot], test_angles, batch_size)
    else:
        batches_per_test = (len(test_set.filenames) + batch_size - 1) // batch_size
        test_metrics = model.evaluate_generator(test_set, batches_per_test)

    print('Test set evaluation:')
    for metric_name, metric_value in zip(model.metrics_names, test_metrics):
        print(metric_name, ':', metric_value)

    return model


# def train(datadir, net_filename):
def train(
        # output files
        net_filename, preproc_filename,
        # dataset params
        dataset_dir, dataset_name=None, dataset_size=None, dataset_static=False, dataset_inmem=False,
        shuffle_train=True, seed=42, image_data_format=K.image_data_format(),
        # model params
        features=Features.TRAIN, img_side='min', resolution_degrees=.5, grayscale=True, preproc='default',
        angle_encoding=AngleEncoding.SINCOS, force_xy=None, bounding=Bounding.NONE, n_classes=None, dropout=None,
        # training params
        batch_size=2, optimizer=sgd, lr=5e-4, optimizer_kwargs={'momentum': .9, 'nesterov': True}, max_epochs=50):
    from estimate_rotation.dataset import datasets
    from estimate_rotation.model import model, preproc as default_preproc

    # if the dataset is static, then we can deduce image size
    if not dataset_static and img_side == 'min':
        if features == Features.TRAIN:
            img_side = int(math.ceil(158. / resolution_degrees))
        elif features == Features.RESNET50:
            img_side = 224
        elif features == Features.VGG16:
            img_side = 224
        elif features == Features.INCEPTIONV3:
            img_side = 224

    if angle_encoding == AngleEncoding.CLASSES and n_classes is None:
        n_classes = int(math.ceil(360. / resolution_degrees))

    if preproc == 'default':
        preproc = default_preproc[features]

    with open(preproc_filename, 'wb') as preproc_file:
        pickle.dump({'grayscale': grayscale, 'preproc': preproc, 'img_side': img_side,
                     'angle_encoding': angle_encoding, 'n_classes': n_classes},
                    preproc_file)

    train_set, val_set, test_set = datasets(dataset_dir, dataset_name, dataset_size, dataset_static, dataset_inmem,
                                            img_side, grayscale, preproc, angle_encoding, n_classes, batch_size,
                                            shuffle_train, seed, image_data_format)

    rot_predictor, frozen_layers = model(features, img_side, grayscale, angle_encoding, force_xy, bounding, n_classes,
                                         dropout, decode_angle=False)

    trained_rot_predictor = do_training(train_set, val_set, test_set, dataset_inmem, batch_size,
                                        rot_predictor, frozen_layers, angle_encoding, n_classes,
                                        optimizer, lr, optimizer_kwargs, max_epochs,
                                        net_filename)

    return trained_rot_predictor


def main():
    # playground for training various models
    from estimate_rotation.dataset import DatasetSize
    from estimate_rotation.model import Features

    assert K.image_data_format() == 'channels_first'

    # user params
    resolution_degrees = .5
    dataset_dir = '/home/bogdan/work/visionsemantics/data/'
    dataset_name = 'coco'
    dataset_size = DatasetSize.TINY
    dataset_static = True
    dataset_inmem = True
    net_filename = '/home/bogdan/work/visionsemantics/models/tiny_net.h5'
    preproc_filename = '/home/bogdan/work/visionsemantics/models/preproc.pkl'
    # ~user params

    # metaparams
    features = Features.TRAIN
    grayscale = True
    angle_encoding = AngleEncoding.SINCOS
    force_xy = None
    n_classes = None  # can override default n_classes deduced from resolution_degrees
    bounding = Bounding.NONE
    dropout = None

    batch_size = 2
    max_epochs = 50
    learning_rate = 1e-4
    # optimizer = nadam
    optimizer = sgd
    # ~metaparams

    optimizer_kwargs = {}
    if optimizer == sgd:
        optimizer_kwargs['momentum'] = .9
        optimizer_kwargs['nesterov'] = True

    min_img_side = int(math.ceil(158. / resolution_degrees))
    img_side = min_img_side

    train(net_filename, preproc_filename, dataset_dir, dataset_name, dataset_size, dataset_static, dataset_inmem,
          features=features, img_side=img_side, resolution_degrees=resolution_degrees, grayscale=grayscale,
          angle_encoding=angle_encoding, force_xy=force_xy, bounding=bounding, n_classes=n_classes, dropout=dropout,
          batch_size=batch_size, optimizer=optimizer, lr=learning_rate, optimizer_kwargs=optimizer_kwargs,
          max_epochs=max_epochs)


if __name__ == '__main__':
    main()

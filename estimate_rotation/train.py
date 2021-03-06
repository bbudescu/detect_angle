import gc
import os
from collections import OrderedDict
import math
import pickle
import warnings
import numpy
import platform

from estimate_rotation import config
config.keras('theano', platform.node())

from keras.optimizers import nadam, sgd
from keras.losses import mse, categorical_crossentropy
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, ModelCheckpoint
from keras.models import load_model

from estimate_rotation.common import AngleEncoding
from estimate_rotation.model import Features, Bounding, to_deg
from estimate_rotation.dataset import merge_datasets


def diff_angles(angles_true, angles_pred, encoding):
    if encoding == AngleEncoding.DEGREES:
        val_range = 360.
    elif encoding == AngleEncoding.RADIANS:
        val_range = 2. * math.pi
    elif encoding == AngleEncoding.UNIT:
        val_range = 2.
    else:
        raise ValueError('only direct encoding supported for angle differences')
    diffs = (angles_true - angles_pred) % val_range  # in 0...360 (floordiff)

    return K.minimum(diffs, val_range - diffs)


def mae_angles(angles_true, angles_pred, encoding):
    return K.mean(K.abs(diff_angles(angles_true, angles_pred, encoding)), axis=-1)


class LossSavingModelCheckpoint(ModelCheckpoint):
    """
    extends keras ModelCheckpoint to also save the loss when it finds the best monitor value
    """
    def __init__(self, *keras_args, **keras_kwargs):
        self.loss_at_best = None
        super(LossSavingModelCheckpoint, self).__init__(*keras_args, **keras_kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if self.epochs_since_last_save + 1 >= self.period and self.save_best_only and logs is not None:
            monitor_val = logs[self.monitor]
            if self.monitor_op(monitor_val, self.best):
                self.loss_at_best = logs['loss']

        super(LossSavingModelCheckpoint, self).on_epoch_end(epoch, logs)


class GoodEnough(Callback):
    """
    callback that stops training when a certain error is reached on the training set; this is used to prevent
    overfitting the training data in the absence of a cross-validation set that could be used for early stopping
    """
    def __init__(self, dataset, batch_size, in_mem, monitor, batch_eval_freq, stop_at, mode='min'):
        assert mode in ('min', 'max')
        super(GoodEnough, self).__init__()
        self.dataset = dataset
        self.in_mem = in_mem
        self.batch_size = batch_size
        self.monitor = monitor
        self.stop_at = stop_at
        self.mode = mode
        self.eval_freq = batch_eval_freq

    def on_batch_end(self, batch, logs=None):
        if batch % self.eval_freq == 0:
            # the training loss in the logs dict argument is an average over batch losses, which is not very
            # precise, as the model changes at every iteration; that's why we prefer to compute the error on the
            # whole training set here
            if self.in_mem:
                inputs, outputs = self.dataset
                logs = self.model.evaluate(inputs, outputs, self.batch_size, verbose=0)
            else:
                logs = self.model.evaluate_generator(self.dataset, len(self.dataset))

            logs = OrderedDict(zip(self.model.metrics_names, logs))
            current = logs.get(self.monitor, None)
            if current is None:
                warnings.warn('Stop at error requires %s available!' % self.monitor, RuntimeWarning)
                self.model.stop_training = True
            else:
                if self.mode == 'max' and current >= self.stop_at or self.mode == 'min' and current <= self.stop_at:
                    self.model.stop_training = True


def training_session(train_set, val_set, data_inmem, batch_size,
                     model, min_epochs, max_epochs,
                     early_stopping, model_checkpoint, lr_annealing):

    if data_inmem:
        [train_img, train_rot], train_angles = train_set
        if min_epochs:
            model.fit([train_img, train_rot], train_angles, batch_size, min_epochs, callbacks=[model_checkpoint],
                      validation_data=val_set, shuffle=True)

        model.fit([train_img, train_rot], train_angles, batch_size, max_epochs - min_epochs,
                  callbacks=[early_stopping, model_checkpoint, lr_annealing], validation_data=val_set, shuffle=True)
    else:
        if min_epochs:
            model.fit_generator(train_set, len(train_set), min_epochs,
                                callbacks=[model_checkpoint],
                                validation_data=val_set, validation_steps=len(val_set) if val_set else None)

        model.fit_generator(train_set, len(train_set), max_epochs - min_epochs,
                            callbacks=[early_stopping, model_checkpoint, lr_annealing],
                            validation_data=val_set, validation_steps=len(val_set) if val_set else None)


def test_session(model, data_inmem, batch_size, test_set):
    if not test_set:
        return

    if data_inmem:
        [test_img, test_rot], test_angles = test_set
        test_metrics = model.evaluate([test_img, test_rot], test_angles, batch_size)
    else:
        test_metrics = model.evaluate_generator(test_set, len(test_set))

    print('Test set evaluation:')
    for metric_name, metric_value in zip(model.metrics_names, test_metrics):
        print(metric_name, ':', metric_value)


def do_training(
        # training data params (in keras batch_size is passed to the dataset, although it's, technically, a training metaparameter)
        train_set, val_set, test_set, data_inmem, batch_size,
        # model params
        model, frozen_layers, angle_encoding,
        # training metaparameters
        optimizer_class, lr, optimizer_kw_args, min_epochs, max_epochs,
        # output
        net_filename, stage_results_filename,
        # helper params for tuning
        stages=(1, 2, 3), retrain=False
):
    optimizer = optimizer_class(lr, **optimizer_kw_args)

    def mae_degrees(angles_true, angles_pred):
        if angle_encoding in {AngleEncoding.SINCOS, AngleEncoding.CLASSES}:
            # we can't infer n_classes from the shape of angles_true, because it is (None, None)
            n_classes = K.int_shape(angles_pred)[-1]
            err = mae_angles(to_deg(angles_true, angle_encoding, n_classes), to_deg(angles_pred, angle_encoding),
                             AngleEncoding.DEGREES)
        elif angle_encoding in {AngleEncoding.DEGREES, AngleEncoding.RADIANS, AngleEncoding.UNIT}:
            err = to_deg(mae_angles(angles_true, angles_pred, angle_encoding), angle_encoding)
        else:
            raise NotImplementedError()

        return err

    def mse_angles(degrees_true, degrees_pred):
        return K.mean(K.square(diff_angles(degrees_true, degrees_pred, angle_encoding)), axis=-1)

    metric = mae_degrees
    metric_name = metric.__name__

    # maybe use resolution_degrees for this
    metric_epsilon = .5  # an improvement of half a degree over an epoch is considered significant enough to keep training

    if angle_encoding == AngleEncoding.CLASSES:
        loss = categorical_crossentropy
        # loss_epsilon = .005  # hardcoded
    else:
        if angle_encoding == AngleEncoding.SINCOS:
            loss = mse
        else:
            loss = mse_angles

        diff_range = {
            # we consider tan ~ linear around 0, that's why we apply the same math for epsilon
            AngleEncoding.SINCOS: 4.,  # vals in [-1, 1] => diffs in [-2, 2]
            AngleEncoding.UNIT: 2.,
            AngleEncoding.RADIANS: 2. * math.pi,
            AngleEncoding.DEGREES: 360
        }

        one_degree_diff = diff_range[angle_encoding] / 360.
        diff_epsilon = one_degree_diff * metric_epsilon
        # loss_epsilon = diff_epsilon ** 2  # we use mean SQUARE error

    train_monitor = val_monitor = metric_name
    if val_set:
        val_monitor = 'val_' + val_monitor

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    reduce_lr_cooldown = 1
    reduce_lr_patience = 2

    early_stopping_patience = (reduce_lr_cooldown + reduce_lr_patience) * 2 + 1

    # @TODO: for a real life scenario: create our own EarlyStopping implementation that supports decisions based on
    #        relative improvement, and that increases patience proportionally with the number of epochs so far; also,
    #        a minimum number of epochs should be allowed as param
    early_stopping = EarlyStopping(val_monitor, min_delta=metric_epsilon, patience=early_stopping_patience, verbose=1)
    model_checkpoint = LossSavingModelCheckpoint(net_filename, val_monitor, verbose=1, save_best_only=True)

    lr_schedule = ReduceLROnPlateau(train_monitor, factor=0.5, patience=reduce_lr_patience, verbose=1, mode='min',
                                    min_delta=metric_epsilon, cooldown=reduce_lr_cooldown, min_lr=1e-12)

    if 1 not in stages:
        retrain = True

    if retrain:
        with open(stage_results_filename, 'rt') as best_file:
            line = best_file.readline()
            model_checkpoint.best = float(line)

            line = best_file.readline()
            model_checkpoint.loss_at_best = float(line)

    if 1 in stages:
        training_session(train_set, val_set, data_inmem, batch_size, model, min_epochs, max_epochs,
                         early_stopping, model_checkpoint, lr_schedule)

        with open(stage_results_filename, 'wt') as best_file:
            best_file.write(str(model_checkpoint.best) + '\n')
            best_file.write(str(model_checkpoint.loss_at_best) + '\n')

    print(val_monitor, ':', model_checkpoint.best)

    # load net with best xval
    model.load_weights(net_filename)

    if 1 in stages:
        test_session(model, data_inmem, batch_size, test_set)

    if frozen_layers and 2 in stages:
        # if using a pretrained net, restart training to tune the features, too
        for frozen_layer in frozen_layers:
            frozen_layer.trainable = True

        old_optimizer = model.optimizer
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        if old_optimizer:
            # doesn't work... getting oom, anyway
            del old_optimizer
            gc.collect()  # shouldn't be necessary

        # # reset learning rate for optimizer (ReduceLROnPlateau callback might have modified it)
        K.set_value(model.optimizer.lr, lr)
        #
        # # we need to recompile the model after thawing the weights
        # model.compile(optimizer=model.optimizer, loss=loss, metrics=[metric])

        training_session(train_set, val_set, data_inmem, batch_size, model, 0, max_epochs, early_stopping,
                         model_checkpoint, lr_schedule)

        with open(stage_results_filename, 'wt') as best_file:
            best_file.write(str(model_checkpoint.best) + '\n')
            best_file.write(str(model_checkpoint.loss_at_best) + '\n')

        # load net with best xval
        model.load_weights(net_filename)

        print('after tuning the feature extractor')
        print(val_monitor, ':', model_checkpoint.best)
        test_session(model, data_inmem, batch_size, test_set)

    # train on both train and xval combined
    if val_set and 3 in stages:
        if data_inmem:
            [train_imgs, train_rots], train_angles = train_set
            [val_imgs, val_rots], val_angles = val_set

            train_imgs = numpy.vstack((train_imgs, val_imgs))
            train_rots = numpy.vstack((train_rots, val_rots))
            # if angle_encoding in {AngleEncoding.RADIANS, AngleEncoding.UNIT, AngleEncoding.DEGREES}:
            #     assert train_angles.ndim == val_angles.ndim == 1
            #     train_angles = numpy.hstack((train_angles, val_angles))
            # else:
            #     train_angles = numpy.vstack((train_angles, val_angles))

            train_angles = numpy.vstack((train_angles, val_angles))

            full_train_set = [train_imgs, train_rots], train_angles
            batches_per_epoch = (len(train_imgs) + batch_size - 1) // batch_size
        else:
            full_train_set = merge_datasets(train_set, val_set)
            batches_per_epoch = len(full_train_set)

        # we check whether we reached our objective 5 times an epoch
        batch_eval_freq = max(batches_per_epoch // 5, 1)
        good_enough = GoodEnough(full_train_set, batch_size, data_inmem, 'loss', batch_eval_freq,
                                 model_checkpoint.loss_at_best)
        early_stopping = EarlyStopping(train_monitor, min_delta=metric_epsilon, patience=early_stopping_patience,
                                       verbose=1)
        lr_schedule = ReduceLROnPlateau(train_monitor, factor=0.5, patience=reduce_lr_patience, verbose=1,
                                        mode='min', min_delta=metric_epsilon, cooldown=reduce_lr_cooldown,
                                        min_lr=1e-12)

        # reset learning rate for optimizer (ReduceLROnPlateau callback might have modified it)
        K.set_value(model.optimizer.lr, lr)

        # @TODO: fix: good_enough is passed as wrong param
        training_session(train_set, None, data_inmem, batch_size, model, 0, max_epochs, early_stopping,
                         model_checkpoint=good_enough, lr_annealing=lr_schedule)

        model.save(net_filename, overwrite=True)

        print('after training on train + xval')
        test_session(model, data_inmem, batch_size, test_set)

    if not stages:
        test_session(model, data_inmem, batch_size, test_set)

    return model, model_checkpoint.best


# def train(datadir, net_filename):
def train(
        # output files
        net_filename, preproc_filename, stage_results_filename,
        # dataset params
        dataset_dir, dataset_name=None, dataset_size=None, dataset_static=False, dataset_inmem=False,
        shuffle_train=True, seed=42, image_data_format=K.image_data_format(), no_xval=False, no_test=False,
        cache_datasets=False,
        # model params
        features=Features.TRAIN, img_side='min', resolution_degrees=.5, grayscale=True, preproc='default',
        angle_encoding=AngleEncoding.SINCOS, force_xy=None, bounding=Bounding.NONE, n_classes=None,
        convs_per_block=2, skip_layer_connections=2, dropout=None, l2_penalty=1e-4,
        # training params
        batch_size=2, optimizer=sgd, lr=5e-4, optimizer_kwargs={'momentum': .9, 'nesterov': True}, min_epochs=20,
        max_epochs=50,
        # manual tuning helper params
        stages=(1, 2, 3), retrain=False, show_predictions=False):
    from estimate_rotation.dataset import datasets
    from estimate_rotation.model import model, preproc as default_preproc

    # if the dataset is static, then we can deduce image size
    if img_side == 'min':
        if features == Features.TRAIN:
            img_side = int(math.ceil(158. / resolution_degrees))
        elif features == Features.RESNET50:
            img_side = 224
            assert not grayscale
        elif features == Features.VGG16:
            img_side = 224
            assert not grayscale
        elif features == Features.INCEPTIONV3:
            img_side = 224
            assert not grayscale

    if angle_encoding == AngleEncoding.CLASSES and n_classes is None:
        n_classes = int(math.ceil(360. / resolution_degrees))

    if preproc == 'default':
        preproc = default_preproc[features]

    with open(preproc_filename, 'wb') as preproc_file:
        pickle.dump({'preproc': preproc, 'angle_encoding': angle_encoding}, preproc_file)

    train_set, val_set, test_set = datasets(dataset_dir, dataset_name, dataset_size, dataset_static, dataset_inmem,
                                            img_side, grayscale, preproc, angle_encoding, n_classes, batch_size,
                                            shuffle_train, seed, image_data_format,
                                            no_xval, no_test, cache_datasets)

    if retrain:
        rot_predictor = load_model(net_filename, compile=False)

        # try to detect which layers have been frozen intentionally vs. which layers can't be trained at all
        frozen_layers = []
        for layer in rot_predictor.layers:
            trainable = layer.trainable  # cache original value
            layer.trainable = True
            if not trainable and layer.trainable_weights:
                frozen_layers.append(layer)
            layer.trainable = trainable  # restore original value
    else:
        rot_predictor, frozen_layers = model(features, img_side, grayscale, angle_encoding, force_xy, bounding,
                                             n_classes, convs_per_block, skip_layer_connections, dropout, l2_penalty,
                                             decode_angle=False)

    model, err = do_training(train_set, val_set, test_set, dataset_inmem, batch_size, rot_predictor, frozen_layers,
                             angle_encoding, optimizer, lr, optimizer_kwargs, min_epochs, max_epochs, net_filename,
                             stage_results_filename, stages, retrain)

    if show_predictions:
        from estimate_rotation.util import decode_angle
        if not dataset_inmem:
            raise NotImplementedError()

        if test_set is not None:
            dataset = test_set
        elif val_set is not None:
            dataset = val_set
        else:
            dataset = train_set

        predictions = model.predict(dataset[0], batch_size, verbose=1)
        ground_truth = dataset[1]
        print('ground truth, prediction, diff')
        for pred, gt in zip(predictions, ground_truth):
            pred = decode_angle(pred, angle_encoding)
            gt = decode_angle(gt, angle_encoding)

            print('%6.2f : %6.2f (%6.2f)' % (gt, pred, abs(gt - pred)))

    return err


def main():
    from estimate_rotation.dataset import DatasetSize
    from estimate_rotation.model import Features

    assert K.image_data_format() == 'channels_first'

    hostname = platform.node()
    assert hostname in {'mirel', 'nicu'}

    # playground for training various models

    # user params
    resolution_degrees = None
    dataset_dir = os.path.expanduser('~/work/visionsemantics/data/')
    dataset_name = 'coco'
    dataset_size = DatasetSize.MEDIUM
    dataset_static = True
    dataset_inmem = True

    models_dir = os.path.expanduser('~/work/visionsemantics/models')
    net_filename = os.path.join(models_dir, 'net.h5')
    preproc_filename = os.path.join(models_dir, 'preproc.pkl')
    stage_results_filename = os.path.join(models_dir, 'stage_resutls.txt')

    show_predictions = True
    cache_datasets = True
    no_xval = False
    no_test = False
    retrain = False
    # stages = (1, 2, 3,)
    stages = tuple()
    # ~user params

    # laptop:
    #     tf:
    #         no_pretrain, resolution_degrees=.5 (img_size=316):
    #             sgd: max batch size = 3
    #             nadam: max batch size = 1
    #         [vgg16, resnet50, inception_v3] (img_size=224), [nadam, sgd]: fails with batch size=1 in stage 1
    #     theano:
    #         no_pretrain, img_size=316:
    #             nadam: max batch size = 5
    #             sgd: max batch size = 7
    #         vgg16 (img_size=224):
    #             sgd: oom with batch size = 1
    #             nadam: oom with batch size = 1
    #         resnet50 (img_size=224):
    #             sgd: oom with batch size = 1
    #             nadam: oom with batch size = 1
    #         inception_v3 (img_size=224):
    #             sgd: oom with batch size = 1
    #             nadam: oom with batch size = 1
    #
    # desktop:
    #     tf: doesn't work at all
    #     theano: not yet explored. We'll just use the same batch sizes as on the laptop, although the desktop has
    #             twice the memory.
    #             not sure, but I think I did a bunch of tests, and vgg16 worked with batch_size = 5 (sgd), and
    #             batch_size = 1 with nadam (first stage only)

    # metaparams
    features = Features.TRAIN
    grayscale = False
    angle_encoding = AngleEncoding.UNIT
    force_xy = True
    n_classes = None
    bounding = Bounding.NONE
    dropout = .1
    l2_penalty = 2e-4
    img_side = 256

    convs_per_block = 1
    skip_layer_connections = False

    min_epochs = 0
    max_epochs = 100

    learning_rate = 2e-3

    optimizer = nadam

    batch_size = 25

    # if K.backend() == 'tensorflow':
    #     if optimizer == nadam:
    #         batch_size = 1
    #     else:
    #         batch_size = 3
    # elif K.backend() == 'theano':
    #     if optimizer == nadam:
    #         batch_size = 3
    #     else:
    #         batch_size = 5

    # ~metaparams

    optimizer_kwargs = {}
    if optimizer == sgd:
        optimizer_kwargs['momentum'] = 0
        optimizer_kwargs['nesterov'] = False

    train(net_filename, preproc_filename, stage_results_filename, dataset_dir, dataset_name, dataset_size,
          dataset_static, dataset_inmem,
          no_xval=no_xval, no_test=no_test, cache_datasets=cache_datasets,
          features=features, img_side=img_side, resolution_degrees=resolution_degrees, grayscale=grayscale,
          angle_encoding=angle_encoding, force_xy=force_xy, bounding=bounding, n_classes=n_classes,
          convs_per_block=convs_per_block, skip_layer_connections=skip_layer_connections, dropout=dropout,
          l2_penalty=l2_penalty,
          batch_size=batch_size, optimizer=optimizer, lr=learning_rate, optimizer_kwargs=optimizer_kwargs,
          min_epochs=min_epochs, max_epochs=max_epochs, stages=stages, retrain=retrain,
          show_predictions=show_predictions)


if __name__ == '__main__':
    main()

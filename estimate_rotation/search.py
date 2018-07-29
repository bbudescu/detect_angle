import numpy
import gc
import os
from math import ceil
from keras.optimizers import nadam, sgd
import shutil
import traceback
try:
    import cPickle as pickle
except ImportError:
    import pickle
from hyperopt.hp import quniform, choice
from hyperopt import STATUS_OK, STATUS_FAIL, fmin, tpe, Trials

from estimate_rotation.model import Features, Bounding
from estimate_rotation.common import AngleEncoding


def fix_args(**kwargs):
    kwargs = kwargs.copy()
    feature_params = kwargs.pop('feature_model_params')
    kwargs.update(feature_params)
    angle_encoding_params = kwargs.pop('angle_encoding_params')
    kwargs.update(angle_encoding_params)

    kwargs['solver'] = kwargs['solver_params'].pop('solver')

    kwargs['img_side'] = ceil(kwargs['img_side'])
    if kwargs['n_classes']:
        kwargs['n_classes'] = int(kwargs['n_classes'])
    kwargs['batch_size'] = int(kwargs['batch_size'])

    return kwargs


def get_space(overfit):
    # resolution_degrees is used only when not using pretrained features, or when angles are encoded as classes

    untrained_features_space = {
        'img_side': 158. / quniform('resolution img size', .5, 2, .25),
        'grayscale': choice('convert to grayscale', [False, True]),
        'features': Features.TRAIN,
    }

    pretrained_features_space = {
        'img_side': 224,
        'grayscale': False,
        'features': choice('pretrained features', [Features.VGG16, Features.RESNET50, Features.INCEPTIONV3])
    }

    features_space = {
        'feature_model_params': choice('pretrained', [untrained_features_space, pretrained_features_space])
    }

    class_encoding_space = {
        'angle_encoding': AngleEncoding.CLASSES,
        'n_classes': 360 * quniform('resolution classes', .5, 2, .5),
        'force_xy': None,
        'bounding': None
    }

    sincos_encoding_space = {
        'angle_encoding': AngleEncoding.SINCOS,
        'n_classes': None,
        'force_xy': None,
        # the targets are bounded to [-1, 1], anyway, so bounding the output actually just bounds the gradient when very
        # large errors are encountered; also, during testing atan2 is applied, which doesn't care about the norm
        'bounding': Bounding.NONE
    }

    direct_encoding_space = {
        'angle_encoding': choice('direct encoding type', [AngleEncoding.UNIT, AngleEncoding.RADIANS,
                                                          AngleEncoding.DEGREES]),
        'n_classes': None,
        'force_xy': True,  # this could also be false, but this helps avoid the modulo 360 problem
        'bounding': choice('penultimate layer bounding mode', [Bounding.NONE, Bounding.NORM, Bounding.TANH,
                                                               Bounding.CLIP])
    }

    encoding_space = {
        'angle_encoding_params': choice('angle encoding', [class_encoding_space, sincos_encoding_space,
                                                           direct_encoding_space])
    }

    model_param_space = {
        'decode_angle': False,
        'dropout': None if overfit else quniform('layer1_dropout_var', 0, 0.7, 0.1),
    }

    model_param_space.update(features_space)
    model_param_space.update(encoding_space)

    training_param_space = {
        # we treat the lr common to both solvers, because nesterov momentum and batch normalization will, probably,
        # enable high learning rates for any solver
        'lr': 2 ** quniform('learning rate', -30, 3, 1),  # learning rates between 1e-9 and 8, exponential step
        'batch_size': 2 ** quniform('batch size', 0, 4, 1),  # batch size in [1, 2, 4, 8, 16]
        'solver_params': choice('solver params', [
            {'solver': nadam},
            {'solver': sgd, 'momentum': quniform('sgd_momentum', 0, .9, .1), 'nesterov': True}
        ])
    }

    # merge the two dicts
    space = model_param_space.copy()
    space.update(training_param_space)

    return space


def safe_save(obj, filename):
    with open(filename + '.tmp', 'wb') as temp_file:
        pickle.dump(obj, temp_file)

    shutil.move(filename + '.tmp', filename)


def best_net(train, xval, overfit, max_epochs, max_evals, trials_filename, net_filename, best_net_filename,
             force=False):
    if os.path.isfile(trials_filename) and not force:
        with open(trials_filename, 'rb') as trials_file:
            trials = pickle.load(trials_file)
    else:
        space = get_space(overfit)

        trials = Trials()

        def objective(args):
            args = fix_args(**args)

            safe_save(trials, trials_filename)

            try:
                err = train_nnet(net_kwargs, inputs_train, outputs_train, inputs_xval, outputs_xval,
                                      net_filename, max_epochs, **train_kwargs)
            except:
                print('model training failed!')
                traceback.print_exc()
                return {'status': STATUS_FAIL}
            else:
                if err < objective.min:
                    # @TODO: copy net_filename to best_net_filename
                    pass
                objective.min = min(err, objective.min)
                print('experiment:', objective.iter, '/', max_evals, '; best err:', objective.min)
                # return {'loss': train_loss_at_best_xval, 'true_loss': -best_corr_xval, 'status': STATUS_OK,
                #                 'model': model}
                return {'loss': err, 'status': STATUS_OK, 'args': args}
            finally:
                objective.iter += 1
                gc.collect()  # try to help free some memory...

        objective.min = numpy.inf
        objective.iter = 1

        # by default, tpe.suggest runs 20 random configurations in the beginning to get a rough map of the space
        # to override this behaviour, use this:
        # algo = lambda *args, **kwargs: tpe.suggest(*args, n_startup_jobs=5,**kwargs)

        algo = tpe.suggest

        fmin(objective, space, algo, max_evals, trials, rstate=numpy.random.RandomState(42))

        safe_save(trials, trials_filename)

    ok_id = []
    for status_id, status in enumerate(trials.statuses()):
        if status == STATUS_OK:
            ok_id.append(status_id)

    losses = trials.losses()
    losses = [numpy.inf if loss is None else loss for loss in losses]
    min_loss = min(losses)
    min_loss_id_losses = losses.index(min_loss)
    min_loss_id = ok_id[min_loss_id_losses]
    best_args = trials.results[min_loss_id]
    model_kwargs = best_args['model_kwargs']
    train_kwargs = best_args['train_kwargs']

    return model_kwargs, train_kwargs


def main():
    # from hyperopt.pyll.stochastic import sample
    #
    # space = get_space(overfit=True)
    #
    # for _ in range(10):
    #     print(fix_params(**sample(space)))


if __name__ == '__main__':
    main()

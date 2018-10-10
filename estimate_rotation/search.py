import numpy
import gc
import os
import platform
from math import ceil

from estimate_rotation import config
config.keras('theano', platform.node())

from keras.optimizers import nadam, sgd
from keras import backend as K
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
from estimate_rotation.train import train
from estimate_rotation.dataset import DatasetSize


def fix_args(**kwargs):
    kwargs = kwargs.copy()
    feature_params = kwargs.pop('feature_model_params')
    kwargs.update(feature_params)
    angle_encoding_params = kwargs.pop('angle_encoding_params')
    kwargs.update(angle_encoding_params)

    kwargs['convs_per_block'] = int(kwargs['convs_per_block'])
    kwargs['optimizer'] = kwargs['optimizer_kwargs'].pop('optimizer')
    kwargs['batch_size'] = int(kwargs['optimizer_kwargs'].pop('batch_size'))
    if kwargs['n_classes']:
        kwargs['n_classes'] = int(ceil(kwargs['n_classes']))
    kwargs['img_side'] = int(ceil(kwargs['img_side']))

    return kwargs


def get_space(overfit):
    # resolution_degrees is used only when not using pretrained features, and when angles are encoded as classes

    hostname = platform.node()  # maybe the desktop can do stuff the laptop can't (more mem)  TODO
    backend = K.backend()

    momentum_space = quniform('sgd_momentum', 0, .9, .1)

    training_param_space = {
        # we treat the lr common to both solvers, because nesterov momentum and batch normalization will, probably,
        # enable high learning rates for any solver
        'lr': 2 ** quniform('learning rate', -46, 3, 1),  # learning rates between 1e-14 and 8, exponential step
    }

    untrained_features_space = {
        'img_side': 158. / quniform('resolution degrees for img_side', .5, 5, .25),
        'grayscale': choice('convert to grayscale', [False, True]),
        'features': Features.TRAIN,
        'optimizer_kwargs': choice('optimizer kwargs', [
            {
                'optimizer': nadam,
                # @TODO: batch_size now depends on model complexity, too
                # in our test, it worked with 5, but from hyperopt, it crashes mode often
                'batch_size': 1 if backend == 'tensorflow' else 2,
            },
            {
                'optimizer': sgd,
                'batch_size': 3 if backend == 'tensorflow' else 5,
                'nesterov': True,
                'momentum': momentum_space
            }
        ]),
        'convs_per_block': quniform('convs per block', 1, 2, 1),
        'skip_layer_connections': choice('use shortcuts', [False, True]),
        'stages': (1,)
    }

    pretrained_features_space = {
        'img_side': 224,
        'grayscale': False,
        # 'features': choice('pretrained features', [Features.VGG16, Features.RESNET50, Features.INCEPTIONV3])
        'features': Features.VGG16,  # only vgg16 can be trained in stage 2; it might be worth, though, trying out the others, too, with frozen layers
        'optimizer_kwargs': {
            'optimizer': sgd,
            'batch_size': 3 if backend == 'tensorflow' else 5,  # if we get here, backend == theano
            'nesterov': True,
            'momentum': momentum_space
        },
        'convs_per_block': None,
        'skip_layer_connections': None,
        'stages': (1, 2)  # TODO: remove stage 2 when using resnet/inception features (but test first on 1060, maybe they work...)
    }

    if backend == 'tensorflow':
        features_space = {
            'feature_model_params': untrained_features_space
        }
    else:
        features_space = {
            'feature_model_params': choice('pretrained', [untrained_features_space, pretrained_features_space])
        }

    class_encoding_space = {
        'angle_encoding': AngleEncoding.CLASSES,
        'n_classes': 360. / quniform('resolution degrees for n_classes', .5, 2, .5),
        'force_xy': None,
        'bounding': None
    }

    sincos_encoding_space = {
        'angle_encoding': AngleEncoding.SINCOS,
        'n_classes': None,
        'force_xy': None,
        # the targets are bounded to [-1, 1], anyway, so bounding the output actually just bounds the gradient when very
        # large errors are encountered; also, during testing atan2 is applied, which doesn't care about the norm
        'bounding': choice('sincos bounding', [Bounding.NONE, Bounding.TANH])
    }

    direct_encoding_space = {
        'angle_encoding': choice('direct encoding type', [AngleEncoding.UNIT, AngleEncoding.RADIANS,
                                                          AngleEncoding.DEGREES]),
        'n_classes': None,
        'force_xy': choice('force xy', [False, True]),
        'bounding': choice('penultimate layer bounding mode', [Bounding.NONE, Bounding.NORM, Bounding.TANH,
                                                               Bounding.CLIP])
    }

    encoding_space = {
        'angle_encoding_params': choice('angle encoding', [class_encoding_space, sincos_encoding_space,
                                                           direct_encoding_space])
    }

    # 'skip_layer_connections' 'l2_penalty'

    model_param_space = {
        # 'decode_angle': False,
        'dropout': None if overfit else quniform('layer1_dropout_var', 0, 0.7, 0.1),
        'l2_penalty': 0 if overfit else 2 ** quniform('l2 penalty', -46, 0, 1),  # between 1e-14 and 1
    }

    model_param_space.update(features_space)
    model_param_space.update(encoding_space)

    # merge the two dicts
    space = model_param_space.copy()
    space.update(training_param_space)

    space['no_test'] = True

    if overfit:
        space['no_xval'] = True
    else:
        space['no_xval'] = False

    return space


def safe_save(obj, filename):
    with open(filename + '.tmp', 'wb') as temp_file:
        pickle.dump(obj, temp_file)

    shutil.move(filename + '.tmp', filename)


# def best_net(train, xval, overfit, max_epochs, max_evals, trials_filename, net_filename, best_net_filename,
#              force=False):
def best_net(
        # output
        best_net_filename, best_preproc_filename, best_params_filename, best_stage_results_filename,
        # search process parameters:
        overfit, max_evals, trials_filename, overwrite_trials,
        temp_net_filename, temp_preproc_filename, temp_stage_results_filename,
        dataset_dir, dataset_name=None, dataset_size=None, dataset_static=False, dataset_inmem=False,
        shuffle_train=True, seed=42, image_data_format=K.image_data_format(), cache_datasets=False,
        preproc='default', min_epochs=3, max_epochs=50, retrain=False):
    # from the search space we get:
    # no_xval, no_test,
    # features, img_side, resolution_degrees, grayscale, angle_encoding, force_xy, bounding, dropout,
    # batch_size, optimizer, lr, optimizer_kwargs

    if os.path.isfile(trials_filename) and not overwrite_trials:
        with open(trials_filename, 'rb') as trials_file:
            trials = pickle.load(trials_file)
    else:
        space = get_space(overfit)

        trials = Trials()

        def objective(args):
            args = fix_args(**args)

            safe_save(trials, trials_filename)

            print('experiment', objective.iter, '/', max_evals, '; best err:', objective.min)
            print('training args:', args)

            try:
                resolution_degrees = None
                err = train(temp_net_filename, temp_preproc_filename, temp_stage_results_filename, dataset_dir,
                            dataset_name, dataset_size, dataset_static, dataset_inmem, shuffle_train, seed,
                            image_data_format, args['no_xval'], args['no_test'], cache_datasets, args['features'],
                            args['img_side'], resolution_degrees, args['grayscale'], preproc, args['angle_encoding'],
                            args['force_xy'], args['bounding'], args['n_classes'], args['convs_per_block'],
                            args['skip_layer_connections'], args['dropout'], args['l2_penalty'], args['batch_size'],
                            args['optimizer'], args['lr'], args['optimizer_kwargs'], min_epochs, max_epochs,
                            args['stages'], retrain)
            except:
                print('model training failed!')
                traceback.print_exc()
                return {'status': STATUS_FAIL}
            else:
                if err < objective.min:
                    shutil.copyfile(temp_net_filename, best_net_filename)
                    shutil.copyfile(temp_preproc_filename, best_preproc_filename)
                    shutil.copyfile(temp_stage_results_filename, best_stage_results_filename)
                    all_args = args.copy()
                    all_args.update({
                        'dataset_dir': dataset_dir,
                        'dataset_name': dataset_name,
                        'dataset_size': dataset_size,
                        'dataset_static': dataset_static,
                        'dataset_inmem': dataset_inmem,
                        'shuffle_train': shuffle_train,
                        'image_data_format': image_data_format,
                        'cache_datasets': cache_datasets,
                        'resolution_degrees': resolution_degrees,
                        'preproc': preproc,
                        'min_epochs': min_epochs,
                        'max_epochs': max_epochs,
                        'retrain': retrain
                    })

                    with open(best_params_filename, 'wb') as best_params_file:
                        pickle.dump(all_args, best_params_file)
                    print('NEW BEST FOUND')
                objective.min = min(err, objective.min)

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

    return best_args


def test():
    from hyperopt.pyll.stochastic import sample

    space = get_space(overfit=True)

    for _ in range(10):
        print(fix_args(**sample(space)))


def main():
    models_dir = os.path.expanduser('~/work/visionsemantics/models')

    best_args_filename = os.path.join(models_dir, 'best_net_train_params.pkl')
    best_net_filename = os.path.join(models_dir, 'best_net.h5')
    best_preproc_filename = os.path.join(models_dir, 'best_preproc.pkl')
    best_stage_results_filename = os.path.join(models_dir, 'best_stage_results.txt')

    trials_filename = os.path.join(models_dir, 'trials.pkl')

    temp_net_filename = os.path.join(models_dir, 'temp_net.h5')
    temp_preproc_filename = os.path.join(models_dir, 'temp_preproc.pkl')
    temp_stage_results_filename = os.path.join(models_dir, 'temp_stage_results.txt')

    image_data_format = 'channels_first'
    shuffle_train = True
    seed = 42

    dataset_dir = os.path.expanduser('~/work/visionsemantics/data/')
    dataset_name = 'coco'
    dataset_size = DatasetSize.MEDIUM
    dataset_static = True
    dataset_inmem = True
    cache_dataset = True
    preproc = 'default'

    min_epochs = 1
    max_epochs = 10

    retrain = False

    overfit = False
    max_evals = 200
    overwrite_trials = True

    best_args = best_net(best_net_filename, best_preproc_filename, best_args_filename, best_stage_results_filename,
                         overfit, max_evals, trials_filename, overwrite_trials, temp_net_filename,
                         temp_preproc_filename, temp_stage_results_filename, dataset_dir, dataset_name, dataset_size,
                         dataset_static, dataset_inmem, shuffle_train, seed, image_data_format, cache_dataset, preproc,
                         min_epochs, max_epochs, retrain)

    print('best:')
    print(best_args)


if __name__ == '__main__':
    main()

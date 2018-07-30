#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
from collections import OrderedDict

_config_called = False


def keras(backend=None, machine=None):
    # @NOTE: apparently, even tensorflow performance is better when using NCDHW (bc012)
    global _config_called
    # this multi-call check does not work in some cases of circular import, when there are two instances of the function
    # and flag
    if _config_called:
        return False

    _config_called = True
    # config checks
    keras_backends = ('theano', 'tensorflow')
    if backend not in keras_backends:
        raise ValueError('invalid keras backend name')
    machines = {'nicu', 'mirel'}
    if machine not in machines:
        raise ValueError('invalid machine name')

    if backend == 'tensorflow':
        session = tensorflow(verbose=False, profile=False)
    elif backend == 'theano':
        testing_theano = False
        theano_profile = False
        base_compiledir = None

        if machine == 'nicu':
            cuda_root = '/opt/cuda/'
            # cudnn_include_path = '/opt/cudnn6/include/'
            # cudnn_lib_path = '/opt/cudnn6/lib64/'
            cudnn_include_path = os.path.join(cuda_root, 'include')
            cudnn_lib_path = os.path.join(cuda_root, 'lib64')
            preallocate = 0.95
        else:
            cuda_root = '/usr/local/cuda'
            cudnn_include_path = '/usr/local/cuda/include'
            cudnn_lib_path = '/usr/local/cuda/lib64'
            preallocate = 0.75

        device = 'cuda'
        cxx = 'g++'
        optimizations = ('cudnn', 'local_ultra_fast_sigmoid')
        dnn = True

        theano(testing_theano, theano_profile, cuda_root, cudnn_include_path, cudnn_lib_path, preallocate, device, cxx,
               optimizations, dnn, base_compiledir)
    else:
        raise Exception('huh?')

    os.environ['KERAS_BACKEND'] = backend

    from keras import backend as K
    current_keras_backend = K.backend()
    if current_keras_backend != backend:
        raise EnvironmentError('Setting the backend using the environment variable did not work. Did you import keras'
                               'before this call to config.keras?')
    if backend == 'tensorflow':
        K.set_session(session)

    return True


def theano(testing=False, profile=False,
           cuda_root='/opt/cuda', cudnn_include_path='/opt/cuda/include/', cudnn_lib_path='/opt/cuda/lib64/',
           preallocate=0.95, device='cuda', cxx='g++', optimizations=('cudnn', 'local_ultra_fast_sigmoid'), dnn=True,
           base_compiledir=None):
    if testing:
        test_device = device
        device = 'cpu'

    if cuda_root or cudnn_lib_path:
        paths = []
        if cuda_root:
            paths.append(os.path.join(cuda_root, 'lib64'))

        if cudnn_lib_path:
            paths.append(cudnn_lib_path)

        if 'LD_LIBRARY_PATH' in os.environ:
            ld_paths = paths + [os.environ['LD_LIBRARY_PATH']]
        else:
            ld_paths = paths

        os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(ld_paths)

        if 'LIBRARY_PATH' in os.environ:
            lib_paths = paths + [os.environ['LIBRARY_PATH']]
        else:
            lib_paths = paths

        os.environ['LIBRARY_PATH'] = os.pathsep.join(lib_paths)

        paths = []

        if cuda_root:
            paths.append(os.path.join(cuda_root, 'include'))

        if cudnn_include_path:
            paths.append(cudnn_include_path)

        if 'CPATH' in os.environ:
            cpaths = paths + [os.environ['CPATH']]
        else:
            cpaths = paths

        os.environ['CPATH'] = os.pathsep.join(cpaths)

    theano_env_flags = OrderedDict()

    if base_compiledir:
        theano_env_flags['base_compiledir'] = base_compiledir

    theano_env_flags['optimizer_including'] = ':'.join(optimizations)
    theano_env_flags['device'] = device
    theano_env_flags['force_device'] = True
    theano_env_flags['assert_no_cpu_op'] = 'raise'
    theano_env_flags['openmp'] = True
    theano_env_flags['cxx'] = cxx

    if device.startswith('cuda') or (testing and test_device.startswith('cuda')):
        theano_env_flags['gpuarray.preallocate'] = preallocate
    elif device.startswith('gpu') or (testing and test_device.startswith('gpu')):
        theano_env_flags['lib.cnmem'] = preallocate

    if testing:
        theano_env_flags['init_gpu_device'] = test_device

    if cudnn_lib_path:
        theano_env_flags['dnn.library_path'] = cudnn_lib_path

    if cudnn_include_path:
        theano_env_flags['dnn.include_path'] = cudnn_include_path

    if profile:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    theano_env_flags_str = ','.join('='.join(str(kv) for kv in kv_pair) for kv_pair in theano_env_flags.items())
    os.environ['THEANO_FLAGS'] = theano_env_flags_str

    import theano

    theano.config.floatX = 'float32'
    theano.config.on_shape_error = 'raise'
    theano.config.on_opt_error = 'raise'
    theano.config.exception_verbosity = 'high'
    theano.config.allow_gc = False
    # if not testing:
    #     theano.config.warn_float64 = 'warn'

    theano.config.scan.allow_gc = False

    theano.config.gcc.cxxflags = '-D_FORCE_INLINES -std=c++11'

    # disable this so we get warnings in case stuff gets on cpu
    # theano.config.lib.amdlibm = True

    theano.config.dnn.enabled = str(dnn)
    # theano.config.dnn.conv.algo_fwd = 'time_once'  # 'time_on_shape_change'
    theano.config.dnn.conv.algo_fwd = 'none'  # 'time_on_shape_change'

    # CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 is selected if input image is small, but then crashes when getting large images
    # CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 works every time, and it's the default
    # theano.config.dnn.conv.algo_bwd_filter = 'time_once'  # 'time_on_shape_change'
    # theano.config.dnn.conv.algo_bwd_data = 'time_once'  # 'time_on_shape_change'
    theano.config.dnn.conv.algo_bwd_data = 'none'  # 'time_on_shape_change'


def tensorflow(verbose=False, profile=False):
    # if tensorflow_verbose_log:
    #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # this is the default anyway

    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = .95
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF  # options: OFF, ON_1, ON_2
    config.graph_options.infer_shapes = True

    # session = tf.InteractiveSession(config=config)
    session = tf.Session(config=config)

    return session

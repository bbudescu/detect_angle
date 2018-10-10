import argparse
import os
import numpy
import pickle
import platform


def get_args():
    package_name = os.path.basename(os.path.dirname(__file__))
    package_parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    parser = argparse.ArgumentParser(prog=package_name)

    subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands',
                                       help='for subcommand help: %(prog)s <subcommand> -h')
    parser_train = subparsers.add_parser('train', help='retrain the model on a set of images')

    default_data_dir = os.path.join(package_parent_dir, 'data')
    parser_train.add_argument('--datadir', default=default_data_dir,
                              help="path to a directory containing the images on which to train the neural net. It"
                                   "must have subdirectories 'train', 'val' and 'test'. Default is "
                                   "'<parent dir of " + package_name + ">/data'")

    parser_test = subparsers.add_parser('test', help='estimate the angle by which an image was rotated to obtain the '
                                                     'other')
    parser_test.add_argument('img', help='input image')
    parser_test.add_argument('rot_img', help='rotated version of img')

    parser.add_argument('--model_path', default=os.path.join(package_parent_dir, 'net.h5'),
                        help='if training, the path where to store the resulting net; if testing, the model to apply')

    parser.add_argument('--preproc_path', default=os.path.join(package_parent_dir, 'preproc.pkl'),
                        help='if training, the path where to store the resulting net; if testing, the model to apply')

    args = parser.parse_args()

    datadir = args.datadir if 'datadir' in args else None
    img = args.img if 'img' in args else None
    rot = args.rot_img if 'rot_img' in args else None
    net_path = args.model_path
    preproc_path = args.preproc_path

    return net_path, preproc_path, datadir, img, rot


def load_and_preproc(img_filename, grayscale, img_side, preproc):

    from keras.preprocessing.image import load_img
    from estimate_rotation.util import zoom_square

    from keras import backend as K

    img = numpy.asarray(load_img(img_filename, grayscale))

    img = zoom_square(img, img_side)

    if not grayscale and K.image_data_format() == 'channels_first':
        img = numpy.moveaxis(img, 2, 0)
    img = preproc(img)

    if grayscale:
        img = img.reshape((1,) + img.shape)

    return img


def test(net_filename, preproc_filename, img_filename, rot_filename):
    # from estimate_rotation import config
    # config.keras('theano', platform.node())
    from keras.models import load_model
    from estimate_rotation.util import decode_angle
    from estimate_rotation.model import atan2
    from keras import backend as K

    from estimate_rotation.model import model as get_model, Features, Bounding
    from estimate_rotation.common import AngleEncoding

    model, _ = get_model(Features.TRAIN, img_side=256, grayscale=False, angle_encoding=AngleEncoding.UNIT, force_xy=True,
                         bounding=Bounding.NONE, n_classes=None, convs_per_block=1, skip_layer_connections=False,
                         dropout=.1, l2_penalty=2e-4, decode_angle=False)

    model.load_weights(net_filename)
    model.save(net_filename, include_optimizer=False)

    model = load_model(net_filename, compile=False, custom_objects={'atan2': atan2})
    # model.save(net_filename, include_optimizer=False)

    grayscale = model.input_shape[0][1 if K.image_data_format() == 'channels_first' else -1] == 1
    img_side = model.input_shape[0][-2]

    with open(preproc_filename, 'rb') as preproc_file:
        preproc_dict = pickle.load(preproc_file)

    preproc = preproc_dict['preproc']
    angle_encoding = preproc_dict['angle_encoding']

    img = load_and_preproc(img_filename, grayscale, img_side, preproc)
    rot = load_and_preproc(rot_filename, grayscale, img_side, preproc)

    angle = model.predict([img.reshape((1,) + img.shape), rot.reshape((1,) + rot.shape)])
    angle = decode_angle(angle[0], angle_encoding)

    return angle


def main():
    net_filename, preproc_filename, datadir, img_filename, rot_filename = get_args()

    if datadir:
        from estimate_rotation.train import train
        train(net_filename, preproc_filename, datadir)
    else:
        angle = test(net_filename, preproc_filename, img_filename, rot_filename)
        print('estimated rotation angle between the two images:', angle)


if __name__ == '__main__':
    main()

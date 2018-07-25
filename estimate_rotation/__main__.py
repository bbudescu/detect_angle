import argparse
import os
import numpy
import pickle


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
    from scipy.ndimage.interpolation import zoom
    from keras import backend as K

    img = numpy.asarray(load_img(img_filename, grayscale))
    assert img.shape[0] == img.shape[1]
    if img.shape[0] != img_side:
        zoom_level = float(img_side) / img.shape[0]
        if img.ndim == 2:
            img = zoom(img, zoom_level)
        else:
            img = numpy.moveaxis(numpy.stack([zoom(channel, zoom_level) for channel in numpy.moveaxis(img, 2, 0)]), 0, 2)

    if not grayscale and K.image_data_format() == 'channels_first':
        img = numpy.moveaxis(img, 2, 0)
    img = preproc(img)

    if grayscale:
        img = img.reshape((1,) + img.shape)

    return img


def test(net_filename, preproc_filename, img_filename, rot_filename):
    from keras.models import load_model
    from estimate_rotation.util import decode_angle
    from estimate_rotation.train import mean_abs_deg_diff

    with open(preproc_filename, 'rb') as preproc_file:
        preproc_dict = pickle.load(preproc_file)
    grayscale = preproc_dict['grayscale']
    preproc = preproc_dict['preproc']
    img_side = preproc_dict['img_side']
    angle_encoding = preproc_dict['angle_encoding']
    n_classes = preproc_dict['n_classes']

    img = load_and_preproc(img_filename, grayscale, img_side, preproc)
    rot = load_and_preproc(rot_filename, grayscale, img_side, preproc)

    mean_abs_deg_diff.angle_encoding = angle_encoding
    mean_abs_deg_diff.n_classes = n_classes

    model = load_model(net_filename, custom_objects={'mean_abs_deg_diff': mean_abs_deg_diff})
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

from enum import Enum, unique, auto
import numpy
from math import ceil, floor
from glob import glob
import os
from scipy.misc import imsave

from keras.preprocessing.image import Iterator as DatasetIterator, load_img
from keras import backend as K

from estimate_rotation.common import AngleEncoding
from estimate_rotation.util import preprocess_input

@unique
class DatasetSize(Enum):
    TINY = auto()
    SMALL = auto()
    MEDIUM = auto()
    ALL = auto()


def genrot(img, out_img, out_rot, out_format=K.image_data_format()):
    """
    rotates the input image by a random angle, crops a square region with sidelen=min(input_img_width, input_img_height)
    around the centers of both unrotated and rotated versions, and scales them to target_sidelen; random padding is
    added to the rotated image's border/margin regions of unknown values; all random sampling is done with numpy.random
    :param img: input image (channels_last)
    :return: cropped input image, cropped output image, rotation angle
    """
    # as per clarifications from Cheehan:
    # "The padding can be to any arbitrary values (black, white, mirror, border, replicate, etc)."
    # [opencv = numpy.pad = ndimage.rotate] border mode mapping:
    # black = 'constant' = 'constant'
    # white = 'constant' = 'constant'
    # mirror = 'reflect' = 'reflect'
    # border = 'edge' = 'nearest'
    # replicate = 'edge'

    # prev version using numpy.pad
    # mode = numpy.random.choice(['constant', 'reflect', 'edge', 'wrap', 'mean', 'median', 'minimum', 'maximum'])
    mode = numpy.random.choice(['constant', 'nearest', 'reflect', 'wrap'])

    if mode == 'constant':
        cval = numpy.random.choice(['black', 'white', 'random'])

        if numpy.issubdtype(img.dtype, numpy.integer):
            iinfo = numpy.iinfo(img.dtype)
            if cval == 'black':
                cval = iinfo.min
            elif cval == 'white':
                cval = iinfo.max
            else:
                cval = numpy.random.randint(iinfo.min + 1, iinfo.max)
        else:
            assert numpy.issubdtype(img.dtype, numpy.floating)
            assert numpy.all(img >= 0)
            assert numpy.all(img <= 1)
            if cval == 'black':
                cval = 0.
            elif cval == 'white':
                cval = 1.
            else:
                cval = numpy.random.rand()
                while cval == 0.:  # rather unlikely
                    cval = numpy.random.rand()
    else:
        cval = 0.0  # default arg for cval param of scipy.ndimage.interpolation.rotate (ignored, in this case)

    # if no output is specified ndimage.rotate and ndimage.zoom create a new array with the same dtype as the input,
    # which might be something like uint8 or uint16. This will truncate the results, even though they will be later
    # stored in a K.floatx() array.
    if out_img.dtype == K.floatx() and out_rot.dtype == K.floatx():
        img = img.astype(K.floatx())
    else:
        # check out https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#casting-rules for figuring out what
        # the output type should be. Note, however, that according to the table, casting from int to float32 is not
        # safe, although it should be good enough for our purposes (images are rarely stored as uint32, anyway). It also
        # makes sense to keep the data in the original format if output dtype would truncate it.
        # Passing the output directly to transform_affine should fix this issue, and is a better solution then upcasting
        # the inputs.
        raise NotImplementedError('')

    # @TODO: This reallocation can be avoided by fixing util.rot_scale_crop according to the todo note in the beginning
    #        of the function definition

    if img.ndim == 3:
        # convert to channels_first
        img = numpy.moveaxis(img, 2, 0)
        if out_format == 'channels_last':
            # make a transposed view of the array
            out_img = numpy.moveaxis(out_img, 2, 0)
            out_rot = numpy.moveaxis(out_rot, 2, 0)
    else:
        assert img.ndim == 2
        # remove dummy dim
        out_img = out_img[0 if out_format == 'channels_first' else -1]
        out_rot = out_rot[0 if out_format == 'channels_first' else -1]

    rot_deg = float(numpy.random.rand() * 360 - 180)
    zoom_img = float(out_img.shape[1]) / min(img.shape[-2:])
    zoom_rot = float(out_rot.shape[1]) / min(img.shape[-2:])

    from estimate_rotation.util import rot_crop_scale
    rot_crop_scale(img, rot_deg, zoom_img, zoom_rot, mode, cval, out_img, out_rot)

    # TODO: fix rot_scale_crop and use it instead of rot_crop_scale
    # from estimate_rotation.util import rot_scale_crop
    # rot_scale_crop(img, rot_deg, zoom_img, zoom_rot, mode, cval, out_img, out_rot)

    return rot_deg


class DirectoryIterator(DatasetIterator):
    def __init__(self, grayscale, preproc, angle_encoding, n_classes, n, batch_size, shuffle, seed, image_data_format,
                 img_side):
        self.grayscale = grayscale
        self.preproc = preproc
        self.angle_encoding = angle_encoding
        self.data_format = image_data_format

        n_channels = 1 if self.grayscale else 3

        if self.data_format == 'channels_last':
            self.image_shape = (img_side, img_side, n_channels)
        else:
            self.image_shape = (n_channels, img_side, img_side)

        if angle_encoding == AngleEncoding.CLASSES:
            self.n_classes = n_classes
            self.angle_resolution = 2 * numpy.pi / self.n_classes
        else:
            self.angle_resolution = None

        super(DirectoryIterator, self).__init__(n, batch_size, shuffle, seed)

    def allocate_batches(self, batch_size):
        batch_img = numpy.empty((batch_size,) + self.image_shape, dtype=K.floatx())
        batch_rot = numpy.empty_like(batch_img)

        if self.angle_encoding in {AngleEncoding.DEGREES, AngleEncoding.RADIANS, AngleEncoding.UNIT}:
            batch_y = numpy.empty(batch_size, K.floatx())
        elif self.angle_encoding == AngleEncoding.SINCOS:
            batch_y = numpy.empty((batch_size, 2), K.floatx())
        elif self.angle_encoding == AngleEncoding.CLASSES:
            batch_y = numpy.zeros((batch_size, self.n_classes), K.floatx())
        else:
            raise NotImplementedError()

        return batch_img, batch_rot, batch_y

    def proc(self, batch_img, batch_rot, batch_y, angles_deg):
        from estimate_rotation.util import encode_angle
        if self.preproc:
            batch_img[...] = self.preproc(batch_img, self.data_format)
            batch_rot[...] = self.preproc(batch_rot, self.data_format)

        for o, angle_deg in enumerate(angles_deg):
            angle_enc = encode_angle(angle_deg, self.angle_encoding, self.angle_resolution)

            if self.angle_encoding == AngleEncoding.CLASSES:
                batch_y[o, angle_enc] = True
            else:
                batch_y[o] = angle_enc

    def _get_batches_of_transformed_samples(self, index_array):
        raise NotImplementedError()


class AugmentingDirectoryIterator(DirectoryIterator):
    """
    similar to keras.preprocessing.image.DirectoryIterator, but does not assume per-class subdirectories
    """
    def __init__(self, directory, img_sidelen=256, grayscale=False, preproc=preprocess_input,
                 angle_encoding=AngleEncoding.SINCOS, n_classes=720, batch_size=32, shuffle=True, seed=None,
                 image_data_format=K.image_data_format(), outdir=None):

        assert image_data_format in {'channels_first', 'channels_last'}
        assert angle_encoding in AngleEncoding

        self.filenames = sorted(glob(os.path.join(directory, '*')))

        self.outdir = outdir

        if self.outdir:
            self.img_outdir = os.path.join(self.outdir, 'img')
            if not os.path.exists(self.img_outdir):
                os.makedirs(self.img_outdir)

            self.rot_outdir = os.path.join(self.outdir, 'rot')
            if not os.path.exists(self.rot_outdir):
                os.makedirs(self.rot_outdir)

            self.angle_file = open(os.path.join(self.outdir, 'rot.csv'), 'wt')

        super(AugmentingDirectoryIterator, self).__init__(grayscale, preproc, angle_encoding, n_classes,
                                                          len(self.filenames), batch_size, shuffle, seed,
                                                          image_data_format, img_sidelen)

    def __del__(self):
        if self.outdir:
            self.angle_file.close()

    def _get_batches_of_transformed_samples(self, index_array):
        batch_img, batch_rot, batch_y = self.allocate_batches(len(index_array))
        # note: np.random seed is set to initial seed + total_batches_seen at each batch
        angles_deg = []
        for o, i in enumerate(index_array):
            input_filename = self.filenames[i]
            img = numpy.asarray(load_img(input_filename, self.grayscale))
            angle_deg = genrot(img, batch_img[o], batch_rot[o], self.data_format)
            angles_deg.append(angle_deg)

            if self.outdir:
                filename = os.path.basename(input_filename)

                out_img = batch_img[o]
                if self.data_format == 'channels_first':
                    out_img = numpy.moveaxis(out_img, 0, 2)
                imsave(os.path.join(self.img_outdir, filename), out_img)

                out_rot = batch_rot[o]
                if self.data_format == 'channels_first':
                    out_rot = numpy.moveaxis(out_rot, 0, 2)
                imsave(os.path.join(self.rot_outdir, filename), out_rot)

                self.angle_file.write(filename + ', ' + str(angle_deg) + '\n')

        self.proc(batch_img, batch_rot, batch_y, angles_deg)

        return [batch_img, batch_rot], batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class StaticDirectoryIterator(DirectoryIterator):
    """
    it's easier to observe progress when the training set doesn't change at every epoch; we turn on the random
    augmentation only when we observe overfitting
    """
    def __init__(self, directory, preproc, angle_encoding, n_classes, batch_size, shuffle, seed, image_data_format):
        self.imgdir = os.path.join(directory, 'img')
        assert os.path.exists(self.imgdir)
        assert os.path.isdir(self.imgdir)

        self.rotdir = os.path.join(directory, 'rot')
        assert os.path.exists(self.rotdir)
        assert os.path.isdir(self.rotdir)
        
        angle_filename = os.path.join(directory, 'rot.csv')
        assert os.path.exists(angle_filename)
        assert os.path.isfile(angle_filename)

        self.filenames = []
        self.angles = []
        with open(angle_filename, 'rt') as angle_file:
            for line in angle_file.readlines():
                filename, angle = line.split(',')
                self.filenames.append(filename)
                self.angles.append(float(angle))

        assert [os.path.basename(filename) for filename in sorted(glob(os.path.join(self.imgdir, '*')))] == self.filenames
        assert [os.path.basename(filename) for filename in sorted(glob(os.path.join(self.rotdir, '*')))] == self.filenames

        # read an image to get its shape
        img = numpy.asarray(load_img(os.path.join(self.imgdir, self.filenames[0])))

        grayscale = img.ndim == 2

        if not grayscale:
            assert img.ndim == 3
            assert img.shape[2] == 3

        assert img.shape[0] == img.shape[1]

        img_side = len(img)

        super(StaticDirectoryIterator, self).__init__(grayscale, preproc, angle_encoding, n_classes,
                                                      len(self.filenames), batch_size, shuffle, seed, image_data_format,
                                                      img_side)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_img, batch_rot, batch_y = self.allocate_batches(len(index_array))
        angles_deg = []
        for o, i in enumerate(index_array):
            img = numpy.asarray(load_img(os.path.join(self.imgdir, self.filenames[i])))
            rot = numpy.asarray(load_img(os.path.join(self.rotdir, self.filenames[i])))
            if self.data_format == 'channels_first':
                img = numpy.moveaxis(img, 2, 0)
                rot = numpy.moveaxis(rot, 2, 0)
            batch_img[o] = img
            batch_rot[o] = rot
            angles_deg.append(self.angles[i])
        self.proc(batch_img, batch_rot, batch_y, angles_deg)
        return [batch_img, batch_rot], batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def datasets(dataset_dir, dataset_name, dataset_size, dataset_static, dataset_inmem,
             # parameters passed to {Augmenting, Static}DirectoryIterator constructors:
             img_sidelen, grayscale, preproc, angle_encoding, n_classes, batch_size, shuffle, seed, image_data_format):
    from estimate_rotation.common import AngleEncoding
    from estimate_rotation.util import load_as_nparr, encode_angles_inplace, batch_rgb2y

    if dataset_size == DatasetSize.ALL:
        assert not dataset_static
        data_dir = dataset_dir
    else:
        data_dir = dataset_dir
        if dataset_name is not None:
            data_dir = os.path.join(data_dir, dataset_name)
            if dataset_size is not None:
                data_dir += '_' + dataset_size.name.lower()
            if dataset_static:
                data_dir += '_static'

    if dataset_inmem:
        assert dataset_static

    if dataset_static:
        if dataset_inmem:
            if dataset_size in {DatasetSize.TINY, DatasetSize.SMALL}:
                train_img = numpy.load(os.path.join(data_dir, 'train_img.npy'))
                train_rot = numpy.load(os.path.join(data_dir, 'train_rot.npy'))
                train_angles_deg = numpy.load(os.path.join(data_dir, 'train_angles.npy'))

                val_img = numpy.load(os.path.join(data_dir, 'val_img.npy'))
                val_rot = numpy.load(os.path.join(data_dir, 'val_rot.npy'))
                val_angles_deg = numpy.load(os.path.join(data_dir, 'val_angles.npy'))

                test_img = numpy.load(os.path.join(data_dir, 'test_img.npy'))
                test_rot = numpy.load(os.path.join(data_dir, 'test_rot.npy'))
                test_angles_deg = numpy.load(os.path.join(data_dir, 'test_angles.npy'))
            elif dataset_size == dataset_size.MEDIUM:
                # huge npy files take longer to load than jpegs + processing; medium still fits in memory
                [train_img, train_rot], train_angles_deg = load_as_nparr(os.path.join(data_dir, 'train'))
                [val_img, val_rot], val_angles_deg = load_as_nparr(os.path.join(data_dir, 'val'))
                [test_img, test_rot], test_angles_deg = load_as_nparr(os.path.join(data_dir, 'test'))

            if grayscale:
                train_img = batch_rgb2y(train_img, image_data_format)
                train_rot = batch_rgb2y(train_rot, image_data_format)
                val_img = batch_rgb2y(val_img, image_data_format)
                val_rot = batch_rgb2y(val_rot, image_data_format)
                test_img = batch_rgb2y(test_img, image_data_format)
                test_rot = batch_rgb2y(test_rot, image_data_format)

            if preproc:
                train_img = preproc(train_img)
                train_rot = preproc(train_rot)
                val_img = preproc(val_img)
                val_rot = preproc(val_rot)
                test_img = preproc(test_img)
                test_rot = preproc(test_rot)

            if angle_encoding == AngleEncoding.CLASSES:
                angle_resolution = 2 * numpy.pi / n_classes
            else:
                angle_resolution = None

            train_angles = encode_angles_inplace(train_angles_deg, angle_encoding, angle_resolution)
            val_angles = encode_angles_inplace(val_angles_deg, angle_encoding, angle_resolution)
            test_angles = encode_angles_inplace(test_angles_deg, angle_encoding, angle_resolution)

            train = [train_img, train_rot], train_angles
            val = [val_img, val_rot], val_angles
            test = [test_img, test_rot], test_angles

            # @TODO: use keras' Numpy array iterator for shuffling every epoch
        else:
            if grayscale:
                if preproc:
                    def gray_and_preproc(img):
                        gray = batch_rgb2y(img)
                        return preproc(gray)

                    preproc = gray_and_preproc
                else:
                    preproc = batch_rgb2y

            train = StaticDirectoryIterator(os.path.join(data_dir, 'train'), preproc, angle_encoding,
                                            n_classes, batch_size, shuffle, seed, image_data_format)

            val = StaticDirectoryIterator(os.path.join(data_dir, 'val'), preproc, angle_encoding,
                                          n_classes, batch_size, False, None, image_data_format)

            test = StaticDirectoryIterator(os.path.join(data_dir, 'test'), preproc, angle_encoding,
                                           n_classes, batch_size, False, None, image_data_format)
    else:
        train = AugmentingDirectoryIterator(os.path.join(data_dir, 'train'), img_sidelen, grayscale, preproc,
                                            angle_encoding, n_classes, batch_size, shuffle, seed, image_data_format)

        val = AugmentingDirectoryIterator(os.path.join(data_dir, 'val'), img_sidelen, grayscale, preproc,
                                          angle_encoding, n_classes, batch_size, False, None, image_data_format)

        test = AugmentingDirectoryIterator(os.path.join(data_dir, 'test'), img_sidelen, grayscale, preproc,
                                           angle_encoding, n_classes, batch_size, False, None, image_data_format)

    return train, val, test


def main():
    from estimate_rotation.util import generate_static
    # user params

    train = '/home/bogdan/work/visionsemantics/data/coco_medium/train'
    train_out = '/home/bogdan/work/visionsemantics/data/coco_medium_static/train'

    val = '/home/bogdan/work/visionsemantics/data/coco_medium/val'
    val_out = '/home/bogdan/work/visionsemantics/data/coco_medium_static/val'

    test = '/home/bogdan/work/visionsemantics/data/coco_medium/test'
    test_out = '/home/bogdan/work/visionsemantics/data/coco_medium_static/test'

    resolution_degrees = .5

    # ~user params

    # In order to determine the minimum size of the image that is fed to the neural net, we consider the resolution
    # required by an approach based on image registration. A keypoint detector without subpixel accuracy would require a
    # displacement of at least one pixel in order to disambiguate between the location of the same keypoint before and
    # after rotation. We, thus, formulate the constraint that at least half of the pixels that are contained by both
    # images to have the minimally required displacement when rotating by an angle equal to the desired resolution.
    # The largest margins after rotation (and, thus, the minimum common area) are obtained when rotating by 45 degrees.
    # Then we constrain the length, in pixels, of the circle centered in the image center that covers half of the
    # number of pixels in the minimum common area to be at least the number of resolution_degree intervals that fit in
    # the 360 degree possible rotation range.
    min_img_side = int(ceil(158. / resolution_degrees))

    # metaparams
    grayscale = False
    angle_encoding = AngleEncoding.SINCOS
    batch_size = 64

    # choose a slightly larger image size, as many pixels don't have fully populated neighbourhoods
    img_side = int(ceil(1.2 * min_img_side))

    # ~metaparams

    n_classes = int(ceil(360. / resolution_degrees))

    # generate static datasets
    for imgdir, outdir in zip([train, val, test], [train_out, val_out, test_out]):
        generate_static(imgdir, outdir, img_side, batch_size=10)

    # this is how we would use it for training
    train = AugmentingDirectoryIterator(train, img_side, grayscale, preprocess_input, angle_encoding, n_classes,
                                        batch_size, shuffle=True, seed=None)

    # for batch_img, batch_rot in train:
    #     for img, rot in zip(batch_img, batch_rot):
    #         print(1)


if __name__ == '__main__':
    main()

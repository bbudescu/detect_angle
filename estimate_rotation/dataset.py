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


class ImgProcIterator(DatasetIterator):
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

        super(ImgProcIterator, self).__init__(n, batch_size, shuffle, seed)

    def allocate_batches(self, batch_size):
        batch_img = numpy.empty((batch_size,) + self.image_shape, dtype=K.floatx())
        batch_rot = numpy.empty_like(batch_img)

        if self.angle_encoding in {AngleEncoding.DEGREES, AngleEncoding.RADIANS, AngleEncoding.UNIT}:
            batch_y = numpy.empty((batch_size, 1), K.floatx())
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
                batch_y[o, 0] = angle_enc

    def _get_batches_of_transformed_samples(self, index_array):
        raise NotImplementedError()


class AugmentingFileIterator(ImgProcIterator):
    """
    similar to keras.preprocessing.image.DirectoryIterator, but does not assume per-class subdirectories
    """
    def __init__(self, filenames, img_sidelen=256, grayscale=False, preproc=preprocess_input,
                 angle_encoding=AngleEncoding.SINCOS, n_classes=720, batch_size=32, shuffle=True, seed=None,
                 image_data_format=K.image_data_format(), outdir=None):

        assert image_data_format in {'channels_first', 'channels_last'}
        assert angle_encoding in AngleEncoding

        self.filenames = filenames

        self.outdir = outdir

        if self.outdir:
            self.img_outdir = os.path.join(self.outdir, 'img')
            if not os.path.exists(self.img_outdir):
                os.makedirs(self.img_outdir)

            self.rot_outdir = os.path.join(self.outdir, 'rot')
            if not os.path.exists(self.rot_outdir):
                os.makedirs(self.rot_outdir)

            self.angle_file = open(os.path.join(self.outdir, 'rot.csv'), 'wt')

        super(AugmentingFileIterator, self).__init__(grayscale, preproc, angle_encoding, n_classes, len(self.filenames),
                                                     batch_size, shuffle, seed, image_data_format, img_sidelen)

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


class AugmentingDirectoryIterator(AugmentingFileIterator):
    def __init__(self, directory, *args, **kwargs):
        filenames = sorted(glob(os.path.join(directory, '*')))
        super(AugmentingDirectoryIterator, self).__init__(filenames, *args, **kwargs)


class StaticFileIterator(ImgProcIterator):
    def __init__(self, img_filenames, rot_filenames, angles, preproc, angle_encoding, n_classes, batch_size, shuffle,
                 seed, image_data_format, img_side, grayscale):

        assert len(img_filenames) == len(rot_filenames) == len(angles)

        self.img_filenames = img_filenames
        self.rot_filenames = rot_filenames
        self.angles = angles

        # read an image to get its shape
        file_img = numpy.asarray(load_img(img_filenames[0]))
        file_grayscale = file_img.ndim == 2
        if not file_grayscale:
            assert file_img.ndim == 3
            assert file_img.shape[2] == 3

        assert file_img.shape[0] == file_img.shape[1]

        if file_grayscale:
            assert grayscale, 'you want rgb images, but we have only grayscale'

        super(StaticFileIterator, self).__init__(grayscale, preproc, angle_encoding, n_classes, len(self.img_filenames),
                                                 batch_size, shuffle, seed, image_data_format, img_side)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_img, batch_rot, batch_y = self.allocate_batches(len(index_array))

        batch_img_view = batch_img
        batch_rot_view = batch_rot

        if self.data_format == 'channels_first':
            batch_img_view = numpy.moveaxis(batch_img, 1, 3)
            batch_rot_view = numpy.moveaxis(batch_rot, 1, 3)

        if self.grayscale:
            batch_img_view = batch_img_view[..., 0]
            batch_rot_view = batch_rot_view[..., 0]

        angles_deg = []
        for o, i in enumerate(index_array):
            target_sidelen = batch_img.shape[-2]
            target_size = (target_sidelen, target_sidelen)
            batch_img_view[o] = numpy.asarray(load_img(self.img_filenames[i], self.grayscale, target_size,
                                                       interpolation='bicubic'))
            batch_rot_view[o] = numpy.asarray(load_img(self.rot_filenames[i], self.grayscale, target_size,
                                                       interpolation='bicubic'))
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


class StaticDirectoryIterator(StaticFileIterator):
    """
    it's easier to observe progress when the training set doesn't change at every epoch; we turn on the random
    augmentation only when we observe overfitting
    """
    def __init__(self, directory, preproc, angle_encoding, n_classes, batch_size, shuffle, seed, image_data_format,
                 img_side, grayscale):
        imgdir = os.path.join(directory, 'img')
        assert os.path.exists(imgdir)
        assert os.path.isdir(imgdir)

        rotdir = os.path.join(directory, 'rot')
        assert os.path.exists(rotdir)
        assert os.path.isdir(rotdir)
        
        angle_filename = os.path.join(directory, 'rot.csv')
        assert os.path.exists(angle_filename)
        assert os.path.isfile(angle_filename)

        filenames = []
        angles = []
        with open(angle_filename, 'rt') as angle_file:
            for line in angle_file.readlines():
                filename, angle = line.split(',')
                filenames.append(filename)
                angles.append(float(angle))

        img_filenames = sorted(glob(os.path.join(imgdir, '*')))
        rot_filenames = sorted(glob(os.path.join(rotdir, '*')))

        assert [os.path.basename(filename) for filename in img_filenames] == filenames
        assert [os.path.basename(filename) for filename in rot_filenames] == filenames

        super(StaticDirectoryIterator, self).__init__(img_filenames, rot_filenames, angles, preproc, angle_encoding,
                                                      n_classes, batch_size, shuffle, seed, image_data_format,
                                                      img_side, grayscale)


def merge_datasets(dataset1, dataset2, img_side=None, grayscale=None, preproc=None, angle_encoding=None, n_classes=None,
                   batch_size=None, shuffle=None, seed=None, image_data_format=None):
    assert issubclass(dataset1, ImgProcIterator)
    assert issubclass(dataset2, ImgProcIterator)
    if preproc is None:
        assert dataset1.preproc == dataset2.preproc
        preproc = dataset1.preproc

    if angle_encoding is None:
        assert dataset1.angle_encoding == dataset2.angle_encoding
        angle_encoding = dataset1.angle_encoding

    if angle_encoding == AngleEncoding.CLASSES and n_classes is None:
        assert dataset1.n_classes == dataset2.n_classes
        n_classes = dataset1.n_classes

    if batch_size is None:
        assert dataset1.batch_size == dataset2.batch_size
        batch_size = dataset1.batch_size

    if shuffle is None:
        assert dataset1.shuffle == dataset2.shuffle
        shuffle = dataset1.shuffle

    if shuffle and seed is None:
        assert dataset1.seed == dataset2.seed
        seed = dataset1.seed

    if image_data_format is None:
        assert dataset1.data_format == dataset2.data_format
        image_data_format = dataset1.data_format

    if img_side is None:
        assert dataset1.image_shape[1] == dataset2.image_shape[1]
        img_side = dataset1.image_shape[1]

    if grayscale is None:
        assert dataset1.grayscale == dataset2.grayscale
        grayscale = dataset1.grayscale

    if issubclass(dataset1, AugmentingFileIterator):
        assert issubclass(dataset2, AugmentingFileIterator)
        merged = AugmentingFileIterator(dataset1.filenames + dataset2.filenames, img_side, grayscale, preproc,
                                        angle_encoding, n_classes, batch_size, shuffle, seed, image_data_format)
    else:
        assert issubclass(dataset1, StaticFileIterator)
        assert issubclass(dataset2, StaticFileIterator)

        # @TODO: if not grayscale, make sure that both datasets have rgb images

        merged = StaticFileIterator(dataset1.img_filenames + dataset2.img_filenames,
                                    dataset1.rot_filenames + dataset2.rot_filenames,
                                    dataset1.angles + dataset2.angles,
                                    preproc, angle_encoding, n_classes, batch_size, shuffle, seed, image_data_format,
                                    img_side, grayscale)

    return merged


def get_dataset(data_dir, static, inmem, cache, fold, img_side, grayscale, preproc, angle_encoding, n_classes,
                batch_size, shuffle, seed, image_data_format):
    from estimate_rotation.util import load_as_nparr, encode_angles_inplace
    img_suffix = '_' + fold + '_' + str(img_side)
    if grayscale:
        img_suffix += '_grayscale'
    img_suffix += '.npz'

    img_cache = os.path.join(data_dir, 'imgrot' + img_suffix)
    angles_cache = os.path.join(data_dir, 'angles_' + fold + '.npz')
    cached = os.path.exists(img_cache) and os.path.exists(angles_cache)

    if static:
        if inmem:
            if cache and cached:
                loaded = numpy.load(img_cache)
                img = loaded['img'].astype(K.floatx())
                rot = loaded['rot'].astype(K.floatx())
                loaded = numpy.load(angles_cache)
                angles_deg = loaded['angles']
            else:
                # huge npy files take longer to load than jpegs + processing; medium still fits in memory
                [img, rot], angles_deg = load_as_nparr(os.path.join(data_dir, fold), img_side, grayscale=grayscale)
                if cache:
                    numpy.savez_compressed(img_cache, img=img.astype(numpy.uint8), rot=rot.astype(numpy.uint8))
                    numpy.savez_compressed(angles_cache, angles=angles_deg)

            if preproc:
                img = preproc(img)
                rot = preproc(rot)

            angles = encode_angles_inplace(angles_deg, angle_encoding, n_classes)

            return [img, rot], angles

        return StaticDirectoryIterator(os.path.join(data_dir, fold), preproc, angle_encoding, n_classes, batch_size,
                                       shuffle, seed, image_data_format, img_side, grayscale)

    return AugmentingDirectoryIterator(os.path.join(data_dir, fold), img_side, grayscale, preproc, angle_encoding,
                                       n_classes, batch_size, shuffle, seed, image_data_format)


def datasets(dataset_dir, dataset_name, dataset_size, dataset_static, dataset_inmem,
             # parameters passed to {Augmenting, Static}DirectoryIterator constructors:
             img_sidelen, grayscale, preproc, angle_encoding, n_classes, batch_size, shuffle, seed, image_data_format,
             # params to omit datasets:
             no_xval=False, no_test=False, cache=False):
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

    train = get_dataset(data_dir, dataset_static, dataset_inmem, cache, 'train', img_sidelen, grayscale, preproc,
                        angle_encoding, n_classes, batch_size, shuffle, seed, image_data_format)

    val = None if no_xval else get_dataset(data_dir, dataset_static, dataset_inmem, cache, 'val', img_sidelen,
                                           grayscale, preproc, angle_encoding, n_classes, batch_size, shuffle=False,
                                           seed=None, image_data_format=image_data_format)

    test = None if no_test else get_dataset(data_dir, dataset_static, dataset_inmem, cache, 'test', img_sidelen,
                                            grayscale, preproc, angle_encoding, n_classes, batch_size, shuffle=False,
                                            seed=None, image_data_format=image_data_format)

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

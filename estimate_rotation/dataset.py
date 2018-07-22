import numpy
from math import ceil, floor
from glob import glob
import os

from keras.preprocessing.image import Iterator as DatasetIterator, load_img
from keras import backend as K

from estimate_rotation.common import AngleEncoding

rgb2y = numpy.array([.299, .587, .114])

imagenet_mean_rgb_01 = numpy.array([0.485, 0.456, 0.406])
imagenet_std_rgb_01 = numpy.array([0.229, 0.224, 0.225])

imagenet_mean_y_01 = numpy.dot(rgb2y, imagenet_mean_rgb_01)

# we don't know the covariances of r, g and b across the dataset, so we consider them 0, and hope for a decent
# approximation
imagenet_std_y_01 = numpy.linalg.norm(imagenet_std_rgb_01 * rgb2y)


def custom_preproc(img, image_data_format=K.image_data_format()):
    # it would be nice to apply per-image mean-subtraction and standardization, but we'd have to recompute the mean and
    # std for normalized images, which is pretty time consuming, so, for the moment, we're using the mean and std from
    # imagenet and hope they are close to our images' statistics
    # TODO: compute per channel mean and std of pixels across entire dataset after per-image standardization

    # we'll probably use relu activations, so He is better than Xavier. In the He init scheme, inputs are assumed to be
    # the distributed according to the positive part of a gaussian with mean 0 and std 1. That's why we prefer inputs
    # between 0 and 1, rather than -1 and 1 (which are good for sigmoid + xavier)
    if img.ndim == 2:
        mean = imagenet_mean_y_01
        std = imagenet_std_y_01
    else:
        assert img.ndim == 3
        channel_axis = 0 if image_data_format == 'channels_first' else -1
        assert img.shape[channel_axis] == 3
        mean_shape = [1, 1, 1]
        mean_shape[channel_axis] = 3

        mean = imagenet_mean_rgb_01.reshape(mean_shape)
        std = imagenet_std_rgb_01.reshape(mean_shape)

    return (img / 255. - mean) / std


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
    """
    similar to keras.preprocessing.image.DirectoryIterator, but does not assume per-class subdirectories
    """
    def __init__(self, directory, img_sidelen=256, grayscale=False, preproc=custom_preproc,
                 angle_encoding=AngleEncoding.SINCOS, n_classes=720, batch_size=32,  shuffle=True, seed=None,
                 image_data_format=K.image_data_format()):

        assert image_data_format in {'channels_first', 'channels_last'}
        assert angle_encoding in AngleEncoding

        self.data_format = image_data_format
        self.filenames = glob(os.path.join(directory, '*'))
        self.grayscale = grayscale
        self.preproc = preproc
        self.angle_encoding = angle_encoding

        if angle_encoding == AngleEncoding.CLASSES:
            self.n_classes = n_classes
            self.angle_resolution = 2 * numpy.pi / self.n_classes

        n_channels = 1 if self.grayscale else 3

        if self.data_format == 'channels_last':
            self.image_shape = (img_sidelen, img_sidelen, n_channels)
        else:
            self.image_shape = (n_channels, img_sidelen, img_sidelen)

        super(DirectoryIterator, self).__init__(len(self.filenames), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        # note: np.random seed is set to initial seed + total_batches_seen at each batch
        batch_img = numpy.empty((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_rot = numpy.empty_like(batch_img)

        if self.angle_encoding in {AngleEncoding.DEGREES, AngleEncoding.RADIANS, AngleEncoding.UNIT}:
            batch_y = numpy.empty(len(index_array), K.floatx())
        elif self.angle_encoding == AngleEncoding.SINCOS:
            batch_y = numpy.empty((len(index_array), 2), K.floatx())
        elif self.angle_encoding == AngleEncoding.CLASSES:
            batch_y = numpy.zeros((len(index_array), self.n_classes), K.floatx())
        else:
            raise NotImplementedError()

        for o, i in enumerate(index_array):
            img = numpy.asarray(load_img(self.filenames[i], self.grayscale))
            if self.preproc:
                img = self.preproc(img, 'channels_last')

            angle_deg = genrot(img, batch_img[o], batch_rot[o], self.data_format)

            angle_rad = angle_deg * numpy.pi / 180

            if self.angle_encoding == AngleEncoding.DEGREES:
                batch_y[o] = angle_deg
            elif self.angle_encoding == AngleEncoding.RADIANS:
                batch_y[o] = angle_rad
            elif self.angle_encoding == AngleEncoding.UNIT:
                batch_y[o] = angle_deg / 180
            elif self.angle_encoding == AngleEncoding.SINCOS:
                batch_y[o] = numpy.sin(angle_rad), numpy.cos(angle_rad)
            elif self.angle_encoding == AngleEncoding.CLASSES:
                batch_y[o, int(floor((angle_rad + numpy.pi) / self.angle_resolution))] = True
            else:
                raise NotImplementedError()

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


def main():
    # user params

    train = '/home/bogdan/work/visionsemantics/data/coco_subset/train'
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

    train = DirectoryIterator(train, img_side, grayscale, angle_encoding=angle_encoding, n_classes=n_classes,
                              batch_size=batch_size, seed=42)

    for batch_img, batch_rot in train:
        for img, rot in zip(batch_img, batch_rot):
            print(1)


if __name__ == '__main__':
    main()

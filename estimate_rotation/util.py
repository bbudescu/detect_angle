import numpy
from scipy.ndimage.interpolation import zoom, rotate as imrot
from math import floor

from keras.preprocessing.image import apply_affine_transform
from keras import backend as K


def show(img, rot):
    debug = False

    if not debug:
        return

    from matplotlib import pyplot as plt
    dbg_img = img.copy()
    dbg_rot = rot.copy()
    if dbg_img.ndim == 3:
        dbg_img = numpy.moveaxis(dbg_img, 0, 2)
        dbg_rot = numpy.moveaxis(dbg_rot, 0, 2)

    center_row = dbg_img.shape[0] // 2
    center_col = dbg_img.shape[1] // 2
    crosshair_size = 20

    dbg_img[center_row, center_col - crosshair_size // 2: center_col + crosshair_size // 2] = 255
    dbg_img[center_row - crosshair_size // 2: center_row + crosshair_size // 2, center_col] = 255

    dbg_rot[center_row, center_col - crosshair_size // 2: center_col + crosshair_size // 2] = 255
    dbg_rot[center_row - crosshair_size // 2: center_row + crosshair_size // 2, center_col] = 255

    if img.ndim == 2:
        cmap = 'gray'
    else:
        cmap = None
        dbg_img /= 255
        dbg_rot /= 255

    plt.imshow(dbg_img, cmap, interpolation='none')
    plt.figure()
    plt.imshow(dbg_rot, cmap, interpolation='none')
    plt.show()


def crop_center_square(img, patch_side=None):
    """
    :param img: channel_first, if rgb
    :param patch_side:
    :return:
    """
    selector = [slice(None)] * img.ndim

    if patch_side is None:
        # return largest square enclosed in image
        short_axis, long_axis = (-1, -2) if img.shape[-1] < img.shape[-2] else (-2, -1)
        patch_side = img.shape[short_axis]

    assert patch_side <= img.shape[-1]
    assert patch_side <= img.shape[-2]

    for axis in (-2, -1):
        start = (img.shape[axis] - patch_side) // 2
        selector[axis] = slice(start, start + patch_side)

    return img[tuple(selector)]


def rot_crop_scale(img, rot_deg, zoom_img, zoom_rot, mode, cval, out_img, out_rot):
    # initial version, using numpy.pad explicitly:
    # crop_sidelen = min(img.shape[0], img.shape[1])
    # padded_crop_sidelen = int(ceil(crop_sidelen * sqrt(2)))
    # # only pad as much as necessary (i.e., prefer real image data instead of artificial values)
    # # symmetric padding on each of the first two axes
    # padding = [(max(0, int(ceil((padded_crop_sidelen - img.shape[0]) / 2.))),) * 2,
    #            (max(0, int(ceil((padded_crop_sidelen - img.shape[1]) / 2.))),) * 2]
    #
    # if img.ndim == 3:
    #     padding.append((0, 0))
    # else:
    #     # sanity check
    #     assert img.ndim == 2

    #
    # kwargs = {}
    # if mode == 'constant':
    #     kwargs['constant_values'] = cval
    # padded = numpy.pad(img, padding, mode, **kwargs)

    # spline_order = 1 should be bilinear interpolation, but results look kinda crappy
    spline_order = 3

    if rot_deg == 0:
        rot = img
    else:
        if img.ndim == 2:
            rot = imrot(img, rot_deg, reshape=False, order=spline_order, mode=mode, cval=cval)
        else:
            rot = numpy.stack([imrot(channel, rot_deg, reshape=False, order=spline_order, mode=mode, cval=cval)
                               for channel in img])

    show(img, rot)  # for debugging purposes

    # crop
    img = crop_center_square(img)
    rot = crop_center_square(rot)

    # warning: scipy.ndimage.interpolation.zoom sometimes has unexpected behaviour, as per the issue I reported 3 years
    # ago: https://github.com/scipy/scipy/issues/4922. The issue persists to this day, but it's not very important in
    # this case (i.e., it might add some an extra blank line to the output)
    if zoom_img != 1:
        if img.ndim == 2:
            zoom(img, zoom_img, order=spline_order, output=out_img)
        else:
            for in_channel, out_channel in zip(img, out_img):
                zoom(in_channel, zoom_img, order=spline_order, output=out_channel)
    else:
        out_img[...] = img

    if zoom_rot != 1:
        if rot.ndim == 2:
            zoom(rot, zoom_rot, order=spline_order, output=out_rot)
        else:
            for in_channel, out_channel in zip(rot, out_rot):
                zoom(in_channel, zoom_rot, order=spline_order, output=out_channel)
    else:
        out_rot[...] = rot

    show(out_img, out_rot)


def rot_scale_crop(img, rot_deg, zoom_img, zoom_rot, mode, cval, out_img, out_rot):
    # TODO:
    # this doesn't work yet; however, one solution would be to copy apply_affine_transform and modify it to accept and
    # forward the output and output_shape parameters to scipy.ndimage.interpolaton.affine_transform. This would also
    # allow for proper cropping, as well as better performance, and avoiding unnecessary casts on image type without
    # losing any precision (avoid truncations). However, we must compute the shifts for cropping properly. Also, double
    # check the code in NI_GeometricTransform in
    # https://github.com/scipy/scipy/blob/master/scipy/ndimage/src/ni_interpolation.c
    # again, to make sure that outputs are stored as doubles before being cast to output dtype. As far as I can tell
    # after a quick look, this happens, so operations are NOT done in uint8 if input dtype is uint8.
    zoomed = apply_affine_transform(img, theta=0, zx=zoom_img, zy=zoom_img, row_axis=1, col_axis=2, channel_axis=0,
                                    fill_mode=mode, cval=cval)
    out_img[...] = crop_center_square(zoomed, out_img.shape[1:])

    rot = apply_affine_transform(img, theta=rot_deg, zx=zoom_rot, zy=zoom_rot, row_axis=1, col_axis=2, channel_axis=0,
                                 fill_mode=mode, cval=cval)

    out_rot[...] = crop_center_square(rot, out_rot.shape[1])

    show(out_img, out_rot)


rgb2y = numpy.array([.299, .587, .114])

imagenet_mean_rgb_01 = numpy.array([0.485, 0.456, 0.406])
imagenet_std_rgb_01 = numpy.array([0.229, 0.224, 0.225])

imagenet_mean_y_01 = numpy.dot(rgb2y, imagenet_mean_rgb_01)

# we don't know the covariances of r, g and b across the dataset, so we consider them 0, and hope for a decent
# approximation
imagenet_std_y_01 = numpy.linalg.norm(imagenet_std_rgb_01 * rgb2y)


def to_grayscale_inplace(rgb_batch, image_data_format=K.image_data_format()):
    """
    !!! modifies input rgb_batch
    """
    assert rgb_batch.ndim in {3, 4}
    channel_axis = -3 if image_data_format == 'channels_first' else -1
    shape = [1] * rgb_batch.ndim
    shape[channel_axis] = 3
    rgb_batch *= rgb2y.reshape(shape)
    return rgb_batch.sum(axis=channel_axis)


def preprocess_input(img, image_data_format=K.image_data_format()):
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
        channel_axis = -3 if image_data_format == 'channels_first' else -1
        if img.ndim == 4 and img.shape[channel_axis] == 1:
            # we're dealing with a batch of grayscale images
            mean = imagenet_mean_y_01
            std = imagenet_std_y_01
        else:
            assert img.shape[channel_axis] == 3
            mean_shape = [1] * img.ndim
            mean_shape[channel_axis] = 3

            mean = imagenet_mean_rgb_01.reshape(mean_shape)
            std = imagenet_std_rgb_01.reshape(mean_shape)

    return (img / 255. - mean) / std


def encode_angle(deg, angle_encoding, resolution=None):
    from estimate_rotation.common import AngleEncoding
    assert angle_encoding in AngleEncoding
    rad = deg * numpy.pi / 180

    if angle_encoding == AngleEncoding.DEGREES:
        return deg
    elif angle_encoding == AngleEncoding.RADIANS:
        return rad
    elif angle_encoding == AngleEncoding.UNIT:
        return deg / 180.
    elif angle_encoding == AngleEncoding.SINCOS:
        return numpy.sin(rad), numpy.cos(rad)
    elif angle_encoding == AngleEncoding.CLASSES:
        return int(floor((rad + numpy.pi) / resolution))
    else:
        raise NotImplementedError()


def decode_angle(angle_encoded, angle_encoding):
    from estimate_rotation.common import AngleEncoding
    if angle_encoding == AngleEncoding.DEGREES:
        return angle_encoded
    elif angle_encoding == AngleEncoding.RADIANS:
        return angle_encoded * (180 / numpy.pi)
    elif angle_encoding == AngleEncoding.UNIT:
        return angle_encoded * 180
    elif angle_encoding == AngleEncoding.SINCOS:
        return numpy.arctan2(angle_encoded[1], angle_encoded[0]) * (180 / numpy.pi)
    elif angle_encoding == AngleEncoding.CLASSES:
        resolution = 360. / len(angle_encoded)
        return -180 + numpy.argmax(angle_encoded) * resolution + resolution / 2.


def encode_angles_inplace(angles, angle_encoding, resolution=None):
    """
    !!! modifies input angles_deg
    """
    from estimate_rotation.common import AngleEncoding
    assert angle_encoding in AngleEncoding

    if angle_encoding == AngleEncoding.DEGREES:
        return angles

    if angle_encoding == AngleEncoding.UNIT:
        angles /= 180.
        return angles

    # convert to radians
    angles *= numpy.pi / 180

    if angle_encoding == AngleEncoding.RADIANS:
        return angles

    if angle_encoding == AngleEncoding.SINCOS:
        return numpy.vstack((numpy.sin(angles), numpy.cos(angles))).T

    if angle_encoding == AngleEncoding.CLASSES:
        return numpy.floor((angles + numpy.pi) / resolution).astype(int)

    raise NotImplementedError()


def generate_static(imgdir, outdir, img_side, batch_size):
    from estimate_rotation.common import AngleEncoding
    from estimate_rotation.dataset import AugmentingDirectoryIterator
    """
    rotates, zooms and crops all images in imgdir by random rotations and writes them to outdir; it also generates
    a file in outdir containing the angles for each image; choose a batch size that is a divisor of the number of images
    in imgdir.
    """
    ds = AugmentingDirectoryIterator(imgdir, img_side, grayscale=False, angle_encoding=AngleEncoding.DEGREES,
                                     n_classes=1, batch_size=batch_size, shuffle=False, seed=None, outdir=outdir)

    # go once through the whole dataset to write in the dirs
    for batch_idx in range(len(ds)):
        _ = ds[batch_idx]

    ds.angle_file.close()


def load_as_nparr(static_dir):
    from estimate_rotation.common import AngleEncoding
    from estimate_rotation.dataset import StaticDirectoryIterator
    ds = StaticDirectoryIterator(static_dir, preproc=None, angle_encoding=AngleEncoding.DEGREES,
                                 n_classes=None, batch_size=1, shuffle=False, seed=None,
                                 image_data_format=K.image_data_format())
    ds.batch_size = len(ds)
    return ds[0]


def batch_rgb2y(rgb_batch, image_data_format=K.image_data_format()):
    if image_data_format == 'channels_first':
        rgb_batch = numpy.moveaxis(rgb_batch, 1, 3)

    y_batch = rgb_batch.reshape(-1, 3).dot(rgb2y).reshape(rgb_batch.shape[:-1] + (1,))

    if image_data_format == 'channels_first':
        y_batch = numpy.moveaxis(y_batch, 3, 1)
    return y_batch

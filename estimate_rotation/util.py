import numpy
from scipy.ndimage.interpolation import zoom, rotate as imrot

from keras.preprocessing.image import apply_affine_transform


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

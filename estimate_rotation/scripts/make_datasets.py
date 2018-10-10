from math import ceil
import os
import numpy

from estimate_rotation.util import generate_static, load_as_nparr


def main():
    # we start with 3 directories, each of which has subdirectories train, val and test, each of which, in turn contains
    # some images. The tiny dataset contains 15 train images, 5 validation images and 5 test images, the small one,
    # 100, 30 and 30, and the medium one 1000, 300, 300
    # We want to generate images (1 cropped and one rotated + cropped) that can be used as input for our neural net, and
    # save them in another directory. We will generate them only once, in order to avoid random noise during model
    # validation
    # for each input dir, an output dir will be created with the name <input_dir_name>_static, with subdirs train, val
    # and test, each of which will contain:
    # - a subdir img with square images of the requested size
    # - a subdir rot containing rotated versions of the images above
    # - a file rot.csv containing <filename, angle> information on each line
    # each set will also be saved in numpy format
    dirs = [
        # '/home/bogdan/work/visionsemantics/data/coco_tiny',
        # '/home/bogdan/work/visionsemantics/data/coco_small',
        # '/home/bogdan/work/visionsemantics/data/coco_medium'
        '/home/bogdan/work/visionsemantics/data/coco_large'
    ]

    # batch_sizes = [5, 10, 10, 10]
    batch_sizes = [10]

    # we choose the same parameters as during training
    resolution_degrees = .5
    min_img_side = int(ceil(158. / resolution_degrees))
    # img_side = int(ceil(1.2 * min_img_side))
    img_side = min_img_side

    for in_dir, batch_size in zip(dirs, batch_sizes):
        static_dir = in_dir + '_static'
        for fold in ['train', 'val', 'test']:
            in_subdir = os.path.join(in_dir, fold)
            static_subdir = os.path.join(static_dir, fold)
            if not os.path.exists(static_subdir):
                os.makedirs(static_subdir)
            else:
                assert os.path.isdir(static_subdir)

            generate_static(in_subdir, static_subdir, img_side, batch_size)

            # [imgs, rots], angles = load_as_nparr(static_subdir, img_side, grayscale=False)

            # numpy.save(static_subdir + '_img.npy', imgs)
            # numpy.save(static_subdir + '_rot.npy', rots)
            # numpy.save(static_subdir + '_angles.npy', angles)


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import skimage.exposure
import skimage.io
import skimage.transform


# pre-process pipeline- standard
def pre_process_pipeline(inpath, outdir, eq_hist=True, include_parent_name=False, as_gray=False):
    """
    Preprocess images for standard analysis.

    Reads the image from inpath, applies contrast limited adaptive histogram equalization,
    resizes the images to 224 x 224 for VGG16, and then saves image to disk in directory outdir.

    Parameters
    -----------
    :param include_parent_name:
    :param inpath: str or Path object
        path to image to be pre-processed
    :param outdir: str or Path object
        root directory to save image after pre-processing.
    :param eq_hist: bool
        if True, skimage.exposure.equalize_adapthist is applied before resizing.
        (Can be disabled for sensitivity analysis)
    Saved
    -------
    im_preprocessed: image
            image is saved to disk in outdir with the same filename and image format as the original image.


    """
    name = Path(inpath).name  # get the filename of the image
    if include_parent_name:
        name = inpath.parent.name + '_' + name
    # print(name)
    im = skimage.io.imread(inpath, as_gray=as_gray)  # read in image
    im = skimage.img_as_float32(im)  # convert to float representation for histogram equalization
    if eq_hist:
        im = skimage.exposure.equalize_adapthist(im)  # histogram equalization
    im = skimage.transform.resize(im, (224, 224))  # resize to 224 x 224 needed for VGG16
    im = skimage.img_as_ubyte(im)  # convert back to 8 bit grayscale image
    skimage.io.imsave(Path(outdir, name), im)  # save image to disk
    return


def data_preprocess(files, dataset_name, output_dir):
    output_root = Path(output_dir, dataset_name, 'images_preprocessed',
                       'images_histeq_resize')  # place where pre-processed images will be stored
    os.makedirs(output_root, exist_ok=True)  # create output directory if it does not exist
    for file in files:
        pre_process_pipeline(file, output_root, include_parent_name=False)
    # assert len(list(output_root.glob("*.bmp"))) == 1800  # make sure all 1800 images processed

    # pre-process images without histogram equalization
    output_root_resize = Path(output_dir, dataset_name, 'images_preprocessed',
                              'images_resize')  # place where resized images will be stored
    os.makedirs(output_root_resize, exist_ok=True)  # make output directory if it doesn't already exist

    for file in files:
        pre_process_pipeline(file, output_root_resize, eq_hist=False, include_parent_name=False)

    # assert len(list(output_root_resize.glob("*.bmp"))) == 1800  # make sure all 1800 images processed


def get_list_files(dir_path, extensions=['bmp', 'jpg', 'png']):
    root = Path(dir_path)  # directory where original NEU images are stored
    # assert root.is_dir()  # make sure this directory is found
    list_files = []
    for ext in extensions:
        list_files.extend(root.glob('**/*.{}'.format(ext)))
    files = sorted(list_files, key=lambda file: file.name)
    # print(files)
    return files


def mlcc_data_preprocess():
    list_test = get_list_files('/data4T/ntanh/data/mlcc/test')
    print(len(list_test))
    list_train = get_list_files('/data4T/ntanh/data/mlcc/train')
    print(len(list_train))
    list_valid = get_list_files('/data4T/ntanh/data/mlcc/valid')
    print(len(list_valid))
    # print(lists[0].parent.name)
    data_preprocess(list_train, 'train', '../../data/mlcc/')
    data_preprocess(list_test, 'test', '../../data/mlcc/')
    data_preprocess(list_valid, 'valid', '../../data/mlcc/')


if __name__ == '__main__':
    mlcc_data_preprocess()

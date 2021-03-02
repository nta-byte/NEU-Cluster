import numpy as np
import os
from pathlib import Path
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import sys

# sys.path.append('..')
from helper.classification_tools import CustomLabelEncoder

img_root = Path('data', 'images_preprocessed', 'images_histeq_resize')  # directory where images are stored.
assert img_root.is_dir()  # make sure directory exists and is properly found
files = sorted(img_root.glob("*.bmp"))  # returns a list of all of the images in the directory, sorted by filename.

## Shuffle the filenames so they appear randomly in the dataset.
rs = np.random.RandomState(seed=749976)
rs.shuffle(files)

assert len(files) == 1800  # make sure all files are found.
print('first 10 filenames: {}'.format([x.name for x in files[:10]]))


def extract_labels(f): return [x.stem.split('_')[0] for x in f]


labels = extract_labels(files)
print('first 10 labels: {}'.format(labels[:10]))

le = CustomLabelEncoder()
le.fit(labels, sorter=lambda x: x.upper())

labels_int = le.transform(labels[:10])
labels_str = le.inverse_transform(labels_int)

# save the label encoder so it can be used throughout the rest of this study
with open(Path('models', 'label_encoder.pickle'), 'wb') as f:
    pickle.dump(le, f)

print('label encodings: {}'.format(le.mapper))
print('first 10 integer labels: {}'.format(labels_int))
print('first 10 string labels: {}'.format(labels_str))


# %%

def load_images(paths):
    """
    Loads images in the correct format for use with the Keras VGG16 model

    Images are loaded as PIL image objects, converted to numpy array, and then formatted
    with the appropriate VGG16.preprocess_input() function. Note that this only changes
    how the images are represented, it does not change the actual visual properties of the
    images like preprocessing did before.

    Parameters
    ----------
    paths: list(Path)
        list of Paths to each file where the image is stored. Note that the images should
        have the same height, width in pixels so they can be stored in one array.

    Returns
    ----------
    images: ndarray
        n_images x r x c x 3 array of pixel values that is compatible with the Keras model.

    """

    images = [image.load_img(file) for file in paths]  # load images
    # convert images to an array with shape consistent for the vgg16 input
    images = np.asarray([image.img_to_array(img) for img in images])
    # normalizes the pixel values to match the imagenet format (and therefore the pre-trained weights)
    images = preprocess_input(images)

    return images


# %%

images = load_images(files)
assert len(images) == 1800
print(images.shape)

vgg16_path = Path('models', 'VGG16.h5')
if not vgg16_path.is_file():
    vgg16 = keras.applications.VGG16(include_top=True,  # include fully connected layers
                                     weights='imagenet')  # use pre-trained model
    vgg16.save(vgg16_path)  # save model so we don't have to download it everytime

else:
    vgg16 = keras.models.load_model(vgg16_path)  # use saved model

vgg16.summary()


def layer_extractor(model=vgg16, layer='fc1'):
    """
    returns a model that will extract the outputs of *layer* from *model*.

    Parameters
    -------------
    model: keras model
        full model from which intermediate layer will be extracted
    layer: string
        name of layer from which to extract outputs

    Returns
    -------------
    new_model: keras model
        feature extractor model which takes the same inputs as *model* and returns the outputs
        of the intermediate layer specified by *layer* by calling new_model.predict(inputs)
    """
    assert layer in [x.name for x in model.layers]  # make sure the layer exists

    new_model = keras.Model(inputs=vgg16.input, outputs=[vgg16.get_layer(layer).output])
    for x in new_model.layers:
        print(x.name)
    return new_model


# %% md

fc1_extractor = layer_extractor()
fc1 = fc1_extractor.predict(images, verbose=True)

# save results
results = {'filename': files,
           'features': fc1,
           'labels': labels,
           'layer_name': 'fc1'
           }

feature_dir = Path('data', 'features')
os.makedirs(feature_dir, exist_ok=True)
with open(feature_dir / 'VGG16_fc1_features_std.pickle', 'wb') as f:
    pickle.dump(results, f)

print(fc1.shape)

# %% md


for layer in ['fc2', 'block5_pool', 'block5_conv3']:
    print(f'extracting features: {layer}')
    extractor = layer_extractor(layer=layer)  # model to extract features for each layer
    features = extractor.predict(images, verbose=True)  # features extracted by model
    # save the results using the same format as before
    results = {'filename': files,
               'features': features,
               'labels': labels,
               'layer_name': layer}
    with open(feature_dir / 'VGG16_{}_features.pickle'.format(layer), 'wb') as f:
        pickle.dump(results, f)

# %% md


img_root_nohisteq = Path('data', 'images_preprocessed', 'images_resize')
assert img_root_nohisteq.is_dir()
files_noh = sorted(img_root_nohisteq.glob('*'))
rs = np.random.RandomState(seed=3626210179)
rs.shuffle(files_noh)
labels_noh = extract_labels(files_noh)
assert len(files_noh) == 1800
print('first 5 filenames and labels')
print([x.name for x in files_noh[:5]])
print(labels_noh[:5])

# %%

# follow the same process described above to load images, convert to array, and format for vgg16
images_noh = load_images(files_noh)
fc1_noh = fc1_extractor.predict(images_noh, verbose=True)

results = {'filename': files_noh,
           'features': fc1_noh,
           'labels': labels_noh,
           'layer_name': 'fc1 no_histeq'}
with open(feature_dir / 'VGG16_fc1_features_NoHistEQ.pickle', 'wb') as f:
    pickle.dump(results, f)

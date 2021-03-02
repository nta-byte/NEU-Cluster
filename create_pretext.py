import numpy as np
import os
from pathlib import Path
import pickle
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from libs.helper import CustomLabelEncoder

from libs.dataset.preprocess import get_list_files
from create_pretext_pytorch import init

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
# config = tf.ConfigProto(device_count={'GPU': 0})
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)


def load_images(paths):
    images = [image.load_img(file) for file in paths]  # load images
    images = np.asarray([image.img_to_array(img) for img in images])
    images = preprocess_input(images)

    return images


def layer_extractor(model, layer='fc1'):
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
    for x in model.layers:
        print(x.name)

    new_model = keras.Model(inputs=model.input, outputs=[model.get_layer(layer).output])

    return new_model


def extract_labels(f):
    return [x.stem.split('_')[0] for x in f]


def main():
    args, logging = init()
    img_root = Path('data', args.dataset, 'images_preprocessed',
                    'images_histeq_resize')  # directory where images are stored.
    files = get_list_files(img_root)  # returns a list of all of the images in the directory, sorted by filename.
    files = sorted(files)  # returns a list of all of the images in the directory, sorted by filename.
    print(files)
    # ## Shuffle the filenames so they appear randomly in the dataset.
    rs = np.random.RandomState(seed=749976)
    rs.shuffle(files)

    labels = extract_labels(files)
    print('first 10 labels: {}'.format(labels[:10]))

    le = CustomLabelEncoder()
    le.fit(labels, sorter=lambda x: x.upper())

    labels_int = le.transform(labels[:10])
    labels_str = le.inverse_transform(labels_int)

    # save the label encoder so it can be used throughout the rest of this study
    with open(args.le_path, 'wb') as f:
        pickle.dump(le, f)

    print('label encodings: {}'.format(le.mapper))
    print('first 10 integer labels: {}'.format(labels_int))
    print('first 10 string labels: {}'.format(labels_str))
    images = load_images(files)
    # assert len(images) == 1800
    print(images.shape)

    vgg16_path = Path('models', 'VGG16.h5')
    if not vgg16_path.is_file():
        vgg16 = keras.applications.VGG16(include_top=True,  # include fully connected layers
                                         weights='imagenet')  # use pre-trained model
        vgg16.save(vgg16_path)  # save model so we don't have to download it everytime

    else:
        vgg16 = keras.models.load_model(vgg16_path)  # use saved model

    vgg16.summary()
    fc1_extractor = layer_extractor(vgg16)
    fc1 = fc1_extractor.predict(images, verbose=True)

    # save results
    results = {'filename': files,
               'features': fc1,
               'labels': labels,
               'layer_name': 'fc1'
               }

    feature_dir = Path('data', 'features')
    os.makedirs(feature_dir, exist_ok=True)
    with open('data/neu-cls-64/features/VGG16_fc1_features_std.pickle', 'wb') as f:
        pickle.dump(results, f)

    print(fc1.shape)


if __name__ == '__main__':
    # print(tf.version.VERSION)
    # tf.config.list_physical_devices("GPU")
    main()

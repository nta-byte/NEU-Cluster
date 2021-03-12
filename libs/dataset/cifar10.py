import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    cifar_path = '/data4T/ntanh/data/cifar-10-batches-py/data_batch_1'
    print(unpickle(cifar_path))

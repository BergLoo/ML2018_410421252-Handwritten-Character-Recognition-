import struct
import numpy as np
import gzip
import os

PATH = os.path.dirname(os.path.realpath(__file__)) + '/Original data/'

def read_image(file_name):
    with gzip.open(file_name, 'rb') as f:
        buf = f.read()
        index = 0
        magic, images, rows, columns = struct.unpack_from('>IIII',buf,index)
        index += struct.calcsize('>IIII')
        
        image_size = '>' + str(images * rows * columns) + 'B'
        ims = struct.unpack_from(image_size, buf, index)
        
        im_array = np.array(ims).reshape(images, rows, columns)
        return im_array
    
def read_label(file_name):
    with gzip.open(file_name):
        buf = f.read()
        index = 0
        magic, labels = struct.unpack_from('>II',buf,index)
        index += struct.calcsize('>II')
        
        label_size = '>' +str(labels) + 'B'
        labels - struct.unpack_from(label_size,buf,index)
        
        label_array = np.array(labels)
        return label_array

def getData():
    train_x_data = read_image(PATH + 'train-images-idx3-ubyte.gz')
    train_x_data = train_x_data.reshape(train_x_data.shape[0],-1).astype(np.float32)
    train_y_data = read_image(PATH + 'train-labels-idx1-ubyte.gz')
    test_x_data = read_image(PATH + 't10k-images-idx3-ubyte.gz')
    test_x_data = test_x_data.reshape(train_x_data.shape[0],-1).astype(np.float32)
    test_y_data = read_image(PATH + 't10k-labels-idx1-ubyte.gz')
    print(train_x_data.shape)
    print(train_y_data.shape)
    print(test_x_data.shape)
    print(test_y_data.shape)
    return train_x_data, train_y_data, test_x_data, test_y_data

getData()

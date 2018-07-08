import gzip
import struct
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import tensorflow as tf

def read_image(file_name): #for read iamges
    with gzip.open(file_name, 'rb') as f:
        buf = f.read()
        index = 0
        magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
        index += struct.calcsize('>IIII')

        image_size = '>' + str(images*rows*columns) + 'B'
        ims = struct.unpack_from(image_size, buf, index)
        
        im_array = np.array(ims).reshape(images, rows, columns)
        return im_array

def read_label(file_name): #for read labels
    with gzip.open(file_name, 'rb') as f:
        buf = f.read()
        index = 0
        magic, labels = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')
        
        label_size = '>' + str(labels) + 'B'
        labels = struct.unpack_from(label_size, buf, index)

        label_array = np.array(labels)
        return label_array

print ("Start processing MNIST handwritten digits data...")
train_x_data = read_image("MNIST_data/train-images-idx3-ubyte.gz")
train_x_data = train_x_data.reshape(train_x_data.shape[0], -1).astype(np.float32)
train_y_data = read_label("MNIST_data/train-labels-idx1-ubyte.gz")
test_x_data = read_image("MNIST_data/t10k-images-idx3-ubyte.gz")
test_x_data = test_x_data.reshape(test_x_data.shape[0], -1).astype(np.float32)
test_y_data = read_label("MNIST_data/t10k-labels-idx1-ubyte.gz")

train_x_minmax = train_x_data / 255.0
test_x_minmax = test_x_data / 255.0

# evaluate the softmax regression model by sklearn first
eval_sklearn = False
if eval_sklearn:
    print ("Start evaluating softmax regression model by sklearn...")
    reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")
    reg.fit(train_x_minmax, train_y_data)
    np.savetxt('coef_softmax_sklearn.txt', reg.coef_, fmt='%.6f')  # Save coefficients to a text file
    test_y_predict = reg.predict(test_x_minmax)
    print ("Accuracy of test set: %f" % accuracy_score(test_y_data, test_y_predict))

eval_tensorflow = True
batch_gradient = False
if eval_tensorflow:
    print ("Start evaluating softmax regression model by tensorflow...")
    # reformat y into one-hot encoding style
    lb = preprocessing.LabelBinarizer()
    lb.fit(train_y_data)
    train_y_data_trans = lb.transform(train_y_data)
    test_y_data_trans = lb.transform(test_y_data)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    V = tf.matmul(x, W) + b
    y = tf.nn.softmax(V)

    y_ = tf.placeholder(tf.float32, [None, 10])

    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    if batch_gradient:
        for step in range(300):
            sess.run(train, feed_dict={x: train_x_minmax, y_: train_y_data_trans})
            if step % 10 == 0:
                print ("Batch Gradient Descent processing step %d" % step)
        print ("Finally we got the estimated results, take such a long time...")
    else:
        for step in range(1000):
            sample_index = np.random.choice(train_x_minmax.shape[0], 100)
            batch_xs = train_x_minmax[sample_index, :]
            batch_ys = train_y_data_trans[sample_index, :]
            sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
            if step % 100 == 0:
                print ("Stochastic Gradient Descent processing step %d" % step)
    np.savetxt('coef_softmax_tf.txt', np.transpose(sess.run(W)), fmt='%.6f')  # Save coefficients to a text file
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy of test set: %f" % sess.run(accuracy, feed_dict={x: test_x_minmax, y_: test_y_data_trans}))

    
'''  
程式結果如下
Start processing MNIST handwritten digits data...
Start evaluating softmax regression model by tensorflow...
Stochastic Gradient Descent processing step 0
Stochastic Gradient Descent processing step 100
Stochastic Gradient Descent processing step 200
Stochastic Gradient Descent processing step 300
Stochastic Gradient Descent processing step 400
Stochastic Gradient Descent processing step 500
Stochastic Gradient Descent processing step 600
Stochastic Gradient Descent processing step 700
Stochastic Gradient Descent processing step 800
Stochastic Gradient Descent processing step 900
Accuracy of test set: 0.918200
'''

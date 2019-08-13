#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow_zero_out as z;
# Read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[18]:


def train_network(training_data, labels, output, keep_prob=tf.placeholder(tf.float32), init_sess=None, ts=None, quantize=False):
    learning_rate = 1e-4
    steps_number = 100
    batch_size = 100

    # Read data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Define the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))

    # Training step
    train_step = None
    if ts is None:
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    else:
        train_step = ts

    # Accuracy calculation
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Run the training
    sess = None

    if not init_sess:
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
    else:
        sess = init_sess
        a = sess.run(tf.trainable_variables('W_conv1'))
        print("A = " , a[0][0][0])

    for i in range(steps_number):
        # Get the next batch
        input_batch, labels_batch = mnist.train.next_batch(batch_size)

        # Print the accuracy progress on the batch every 100 steps
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={training_data: input_batch, labels: labels_batch, keep_prob: 1.0})
            print("Step %d, training batch accuracy %g %%"%(i, train_accuracy*100))

        # Run the training step
        train_step.run(feed_dict={training_data: input_batch, labels: labels_batch, keep_prob: 0.5})

    print("The end of training!")

    # Evaluate on the test set
    test_accuracy = accuracy.eval(feed_dict={training_data: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0})
    print("Test accuracy after training: %g %%"%(test_accuracy*100))
    # if quantize:
    #     v1 = sess.graph.get_tensor_by_name('W_conv1:0')
    #     w = z.quantizer(v1)
    #     sess.run(tf.assign(v1, w))

    return sess, train_step
    


# In[5]:


def test_network(sess, training_data, labels, output, keep_prob=tf.placeholder(tf.float32)):
    # Accuracy calculation
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Evaluate on the test set
    test_accuracy = accuracy.eval(feed_dict={training_data: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0})
    print("Test accuracy with conv2d: %g %%"%(test_accuracy*100))


# In[6]:


image_size = 28
labels_size = 10
learning_rate = 0.05
steps_number = 1000
batch_size = 100
hidden_size = 1024


# In[7]:


training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
training_images = tf.reshape(training_data, [-1, image_size, image_size, 1])
labels = tf.placeholder(tf.float32, [None, labels_size])


# In[77]:


W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="W_conv1")
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))


# In[80]:


conv1 = tf.nn.relu(tf.nn.conv2d(training_images, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[81]:


W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="W_conv2")
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))


# In[82]:


conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[83]:


pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])


# In[84]:


W_h = tf.Variable(tf.truncated_normal([7 * 7 * 64, hidden_size], stddev=0.1), name="W_h")
b_h = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
hidden = tf.nn.relu(tf.matmul(pool2_flat, W_h) + b_h)


# In[85]:


keep_prob = tf.placeholder(tf.float32)
hidden_drop = tf.nn.dropout(hidden, keep_prob)


# In[86]:


W = tf.Variable(tf.truncated_normal([hidden_size, labels_size], stddev=0.1), name="W")
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))
output = tf.matmul(hidden_drop, W) + b


# In[102]:

# Train & test the network
sess, ts = train_network(training_data, labels, output, keep_prob)
# sess, ts = train_network(training_data, labels, output, keep_prob, sess, ts)


# In[247]:


test_network(sess, training_data, labels, output, keep_prob)

# In[242]:
weights = []
weights.append(sess.run(tf.trainable_variables('W_conv1')))
weights.append(sess.run(tf.trainable_variables('W_conv2')))
weights.append(sess.run(tf.trainable_variables('W_h')))
weights.append(sess.run(tf.trainable_variables('W:0')))

# In[245]:
def build_test_network(training_images, weights, b_conv1, b_conv2, b_h, b, labels):
    image_size = 28
    labels_size = 10
    learning_rate = 0.05
    steps_number = 100
    batch_size = 100
    hidden_size = 1024
    W_conv1 = weights[0][0]
    print("W = ", W_conv1[0][0][0])
    conv3 = tf.nn.relu(z.custom_conv(input=training_images, filter= W_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1)
    # conv3 = z.custom_conv(input=training_images, filter= W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    pool1 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    W_conv2 = weights[1][0]
    conv4 = tf.nn.relu(tf.nn.conv2d(input=pool1, filter=W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    
    W_h = weights[2][0]
    hidden = tf.nn.relu(tf.matmul(pool2_flat, W_h) + b_h)
    
    keep_prob = tf.placeholder(tf.float32)
    hidden_drop = tf.nn.dropout(hidden, keep_prob)
    
    W = weights[3][0]
    output = tf.matmul(hidden_drop, W) + b
    
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Evaluate on the test set
    test_accuracy = accuracy.eval(feed_dict={training_data: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0})

    print("Test accuracy with custom_conv: %g %%"%(test_accuracy*100))

# In[246]:
# while 10:
build_test_network(training_images, weights, b_conv1, b_conv2, b_h, b, labels)
#     # w = w.eval(feed_dict={training_data: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0})
#     # print(w[0][0])
    
#     # sess.run(v1)

#     # print("shape1 = ", w.shape)
#     # print("shape2 = ", W_conv1.shape)

#     # tf.assign(v1, W_conv10)
#     # print(sess.run(v1))   # 2.0

#     # print(sess.run(v1))   # 1.0
#     # W_conv1 = 0
#     sess, ts = train_network(training_data, labels, output, keep_prob, sess, ts, True)

# weights = []
# weights.append(sess.run(tf.trainable_variables('W_conv2')))
# weights.append(sess.run(tf.trainable_variables('W_h')))
# weights.append(sess.run(tf.trainable_variables('W:0')))
# build_test_network(training_images, weights, b_conv1, b_conv2, b_h, b, labels)
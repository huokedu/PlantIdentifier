from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


'''
Image Reading start
'''
# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./images/*.jpg"))

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)
'''
Image Reading End
'''


#CNN Start
def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    
    conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=21,
          kernel_size=[40, 40],
          padding="same",
          activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    
    conv2 = tf.layers.conv2d(
          inputs=input_layer,
          filters=21,
          kernel_size=[40, 40],
          padding="same",
          activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    pool2_flat = tf.reshape(pool2, [-1, 64 * 64 * 21])
    
    dense = tf.layers.dense(inputs=pool2_flat, units=65536, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(inputs=dropout, units=2)
    
    predictions = {
      "classes": tf.argmax(input=logits, axis = 1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth = 21)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    

with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    
    
eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    

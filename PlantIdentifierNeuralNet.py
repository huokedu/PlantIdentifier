import tensorflow as tf
import sys
import numpy as np
import os
import re
import math
from PIL import Image


pathName=[]
speciesFromIndex=[]
species=[]
indexFromSpecies={}
input=[]
# Our application logic will be added here



for root, dirs, files in os.walk("C:\\Users\\shardool\\Desktop\\100 leaves plant species\\data", topdown=False):
    for name in files:
        pathName.append(name)
        replacedName = re.sub(r"^(.+)_.+?$","\\1",name)
        if not speciesFromIndex.__contains__(replacedName):
            speciesFromIndex.append(replacedName)
            indexFromSpecies[replacedName] = len(speciesFromIndex) - 1
        species.append(indexFromSpecies[replacedName])

print(pathName, '\n', species, '\n', speciesFromIndex, '\n', indexFromSpecies)

for x in range(0,len(pathName) - 1):
    dir = "C:\\Users\\shardool\\Desktop\\100 leaves plant species\\data\\" + pathName[x]
    im = Image.open(dir)
    col, row = im.size
    data = np.zeros((row * col, 2))
    pixels = im.load()
    for i in range(row):
        for j in range(col):
            data[i * col + j, :] = i, j

    w = data.size;
    h = data[0].size;
    np.pad(im,((max(0,math.floor((h-w)/2)), max(0,math.ceil((h-w)/2))), (max(0,math.floor((w-h)/2)), max(0,math.ceil((w-h)/2)))), mode='constant', constant_values=0);

def cnn_model_fn(features, labels, mode):
    input_layer = species

    # Convolutional Layer #1
    # Computes Features for the leaves
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=2,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

     #Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=4,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    #Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    #Pool Flattening
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 10 * 10 * 4])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    #70% probability that the element will be kept due to a dropout rate of 0.3
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)
    predictions = {
        # Generate predictions
        "classes": tf.argmax(input=logits, axis=1),
        # Softmax tensor for predictions.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
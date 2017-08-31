import tensorflow as tf

def bn(inputs, is_training, scope, epsilon=1e-5, momentum=0.9):
  return tf.layers.batch_normalization(
    inputs=inputs,
    epsilon=epsilon,
    momentum=momentum,
    is_training=is_training,
    scope=scope
  )

def conv2d(inputs):
  return tf.layers.conv2d(
    inputs=inputs
  )

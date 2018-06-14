import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from enum import Enum

def dense_block(input_node, layers, name, activation=tf.nn.relu, batch_norm_phase=None, last_layer_activation=False,
                detailed_summary=False):
  with tf.variable_scope(name):
    output = input_node
    for i, layer in enumerate(layers):
      if i == len(layers) - 1 and not last_layer_activation:
        output = tf.layers.dense(output, layer)
      else:
        output = tf.layers.dense(output, layer, activation=activation)

        if batch_norm_phase is not None:
          output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=batch_norm_phase)

      if detailed_summary:
        with tf.name_scope("layer_%d_output" % (i + 1)):
          variable_summaries(output)

  return output

def dense(cls, input_layer, shape, dtype=tf.float32, activation=tf.nn.relu, name="dense", detailed_summary=False):
  with tf.variable_scope(name):
    w = tf.get_variable("w", shape=shape, dtype=dtype, initializer=initializers.xavier_initializer())
    b = tf.get_variable("b", shape=shape[1], dtype=dtype, initializer=tf.zeros_initializer())
    out = tf.nn.bias_add(tf.matmul(input_layer, w), b)

    if detailed_summary:
      with tf.name_scope('w'):
        cls.variable_summaries(w)

      with tf.name_scope('b'):
        cls.variable_summaries(b)

      with tf.name_scope('output'):
        cls.variable_summaries(out)

    if activation is not None:
      return activation(out)
    else:
      return out

def variable_summaries(var, name="summaries"):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def huber_loss(x, delta=1.0):
  return tf.where(
    tf.abs(x) < delta,
    tf.square(x) * 0.5,
    delta * (tf.abs(x) - 0.5 * delta)
  )

def create_target_update_ops(model_name, target_model_name, update_rate):
  # inspired by: https://github.com/yukezhu/tensorflow-reinforce/blob/master/rl/neural_q_learner.py
  net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_name)
  target_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_model_name)

  target_update = []
  for v_source, v_target in zip(net_vars, target_net_vars):
    # this is equivalent to target = (1-alpha) * target + alpha * source
    update_op = v_target.assign_sub(update_rate * (v_target - v_source))
    target_update.append(update_op)

  return tf.group(*target_update)

class NeuralNetwork:

  class Type(Enum):
    MLP = 1
    CNN_MLP = 2

  def __init__(self, config, type):
    self.config = config
    self.type = type

  def build(self, input_dim, output_dim, name):

    with tf.variable_scope(name):

      if self.type == self.Type.MLP:
          input_layer = tf.placeholder(tf.float32, shape=(None, input_dim))
          output_layer = dense_block(input_layer, [*self.config["hidden"], output_dim], "dense", batch_norm_phase=self.config["batch_norm"])
          return input_layer, output_layer
      elif self.type == self.Type.CNN_MLP:
        input_layer = tf.placeholder(tf.float32, shape=(None, *input_dim))

        output = input_layer

        if self.config["pool"] is None:
          iter = zip(self.config["conv"], [None] * len(self.config["conv"]))
        else:
          iter = zip(self.config["conv"], self.config["pool"])

        for conv_config in iter:
          output = tf.layers.conv2d(output, conv_config[0]["num_maps"], conv_config[0]["filter_shape"], strides=conv_config[0]["stride"], padding="same", activation=tf.nn.relu)

          if conv_config[1] is not None:
            output = tf.layers.max_pooling2d(output, conv_config[1]["shape"], conv_config[1]["stride"])

        output = tf.reshape(output, [-1, output.get_shape()[1].value * output.get_shape()[2].value * output.get_shape()[3].value])

        output_layer = dense_block(output, [*self.config["hidden"], output_dim], "dense")
        return input_layer, output_layer
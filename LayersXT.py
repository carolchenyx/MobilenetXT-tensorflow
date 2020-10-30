import tensorflow as tf
import tensorflow.contrib.slim as slim
from opt import opt

class LayerProvider:

    def __init__(self, is4Train):

        self.init_xavier = tf.contrib.layers.xavier_initializer()
        self.init_norm = tf.truncated_normal_initializer(stddev=0.01)
        self.init_zero = slim.init_ops.zeros_initializer()
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(0.00004)

        self.is4Train = is4Train

    def max_pool(self, inputs, k_h, k_w, s_h, s_w, name, padding="SAME"):
        return tf.nn.max_pool(inputs,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    def upsample(self, inputs, shape, name):
        return tf.image.resize_bilinear(inputs, shape, name=name)

    def separable_conv(self, input, c_o, k_s, stride, dilation=1, activationFunc=tf.nn.relu6, scope=""):

        with slim.arg_scope([slim.batch_norm],
                            decay=0.999,
                            fused=True,
                            is_training=self.is4Train,
                            activation_fn=activationFunc):
            output = slim.separable_convolution2d(input,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  trainable=self.is4Train,
                                                  depth_multiplier=opt.depth_multiplier,
                                                  kernel_size=[k_s, k_s],
                                                  rate=dilation,
                                                  weights_initializer=self.init_xavier,
                                                  weights_regularizer=self.l2_regularizer,
                                                  biases_initializer=None,
                                                  activation_fn=tf.nn.relu6,
                                                  scope=scope + '_depthwise')

            output = slim.convolution2d(output,
                                        c_o,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        weights_initializer=self.init_xavier,
                                        biases_initializer=self.init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        trainable=self.is4Train,
                                        weights_regularizer=None,
                                        scope=scope + '_pointwise')

        return output

    def pointwise_convolution(self, inputs, channels, scope=""):

        with tf.variable_scope("merge_%s" % scope):
            with slim.arg_scope([slim.batch_norm],
                                decay=0.999,
                                fused=True,
                                is_training=self.is4Train):
                return slim.convolution2d(inputs,
                                          channels,
                                          stride=1,
                                          kernel_size=[1, 1],
                                          activation_fn=None,
                                          weights_initializer=self.init_xavier,
                                          biases_initializer=self.init_zero,
                                          normalizer_fn=slim.batch_norm,
                                          weights_regularizer=None,
                                          scope=scope + '_pointwise',
                                          trainable=self.is4Train
                                          )

    def inverted_bottleneck(self, inputs, up_channel_rate, channels, stride , k_s=3, dilation=1.0, scope=""):

        with tf.variable_scope("inverted_bottleneck_%s" % scope):
            with slim.arg_scope([slim.batch_norm],
                                decay=0.999,
                                fused=True,
                                is_training=self.is4Train):
                #stride = 2 if subsample else 1

                output = slim.convolution2d(inputs,
                                            up_channel_rate * inputs.get_shape().as_list()[-1],
                                            stride=1,
                                            kernel_size=[1, 1],
                                            weights_initializer=self.init_xavier,
                                            biases_initializer=self.init_zero,
                                            activation_fn=tf.nn.relu6,
                                            normalizer_fn=slim.batch_norm,
                                            weights_regularizer=None,
                                            scope=scope + '_up_pointwise',
                                            trainable=self.is4Train)
                #dw
                output = slim.separable_convolution2d(output,
                                                      num_outputs=None,
                                                      stride=1,
                                                      depth_multiplier=opt.depth_multiplier,
                                                      activation_fn=tf.nn.relu6,
                                                      kernel_size=k_s,
                                                      weights_initializer=self.init_xavier,
                                                      weights_regularizer=self.l2_regularizer,
                                                      biases_initializer=None,
                                                      normalizer_fn=slim.batch_norm,
                                                      rate=dilation,
                                                      padding="SAME",
                                                      scope=scope + '_depthwise_first',
                                                      trainable=self.is4Train)

                #pW-linear
                output = _conv_1x1_bn(output, channels, name="pw_linear", use_bias=True)
                #pw
                output = slim.convolution2d(output,
                                            channels,
                                            stride=1,
                                            kernel_size=[1, 1],
                                            padding='SAME',
                                            activation_fn=tf.nn.relu6,
                                            weights_initializer=self.init_xavier,
                                            biases_initializer=self.init_zero,
                                            normalizer_fn=slim.batch_norm,
                                            weights_regularizer=None,
                                            scope=scope + '_pointwise_inside',
                                            trainable=self.is4Train)#done

                #dw-linear
                output = _dwise_conv(output, k_h=3, k_w=3, depth_multiplier=opt.depth_multiplier, strides=stride,
                            padding='SAME', name='dwise_conv', use_bias=True,
                            reuse=None)


                output = slim.convolution2d(output,
                                            channels,
                                            stride=1,
                                            kernel_size=[1, 1],
                                            activation_fn=None,
                                            weights_initializer=self.init_xavier,
                                            biases_initializer=self.init_zero,
                                            normalizer_fn=slim.batch_norm,
                                            weights_regularizer=None,
                                            scope=scope + '_pointwise',
                                            trainable=self.is4Train)

                if inputs.get_shape().as_list()[1:] == output.get_shape().as_list()[1:]:
                    output = tf.add(inputs, output)
                print("")
        return output

    def convb(self, input, k_h, k_w, c_o, stride, name, relu=True):

        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=True, is_training=self.is4Train):
            output = slim.convolution2d(
                inputs=input,
                num_outputs=c_o,
                kernel_size=[k_h, k_w],
                stride=stride,
                normalizer_fn=slim.batch_norm,
                weights_regularizer=self.l2_regularizer,
                weights_initializer=self.init_xavier,
                biases_initializer=self.init_zero,
                activation_fn=tf.nn.relu if relu else None,
                scope=name,
                trainable=self.is4Train)

        return output

    def stage(self, inputs, outputSize, stageNumber, kernel_size=3):

        output = slim.stack(inputs, self.inverted_bottleneck,
                            [
                                (2, 32, 0, kernel_size, 4),
                                (2, 32, 0, kernel_size, 2),
                                (2, 32, 0, kernel_size, 1),
                            ], scope="stage_%d_mv2" % stageNumber)

        return slim.stack(output, self.separable_conv,
                          [
                              (64, 1, 1),
                              (outputSize, 1, 1)
                          ], scope="stage_%d_mv1" % stageNumber)

    def get(self, input, layerDesc, name):

        if layerDesc['op'] == 'conv2d':
            return self.convb(input, 3, 3, layerDesc['outputSize'], layerDesc['stride'], name, relu=True)
        elif layerDesc['op'] == 'bottleneck':
            return self.inverted_bottleneck(input, layerDesc['expansion'], layerDesc['outputSize'],
                                            layerDesc['stride'] == 2, k_s=3, dilation=layerDesc['dilation'], scope=name)
        elif layerDesc['op'] == 'multi_scale_bottleneck':
            return self.multi_scale_inverted_bottleneck(input, layerDesc['expansion'], layerDesc['outputSize'],
                                                        layerDesc['stride'] == 2, k_s=3, scope=name)
        else:
            return None

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _batch_normalization_layer(inputs, momentum=0.997, epsilon=1e-3, is_training=True, name='bn', reuse=None):
    return tf.layers.batch_normalization(inputs=inputs,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         center=True,
                                         training=is_training,
                                         name=name,
                                         reuse=reuse)


def _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=False, strides=1, reuse=None, padding="SAME"):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
        padding=padding, #('SAME' if strides == 1 else 'VALID'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name,
        reuse=reuse)
    return conv


def _conv_1x1_bn(inputs, filters_num, name, use_bias=True, is_training=True, reuse=None):
    kernel_size = 1
    strides = 1
    x = _conv2d_layer(inputs, filters_num, kernel_size, name=name + "/conv", use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, is_training=is_training, name=name + '/bn',
                                   reuse=reuse)
    return x

def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)

def _conv_bn_relu(inputs, filters_num, kernel_size, name, use_bias=True, strides=1, is_training=True, activation=relu6, reuse=None):
    x = _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, is_training=is_training, name=name + '/bn',
                                   reuse=reuse)
    x = activation(x)
    return x


def _dwise_conv(inputs, k_h=3, k_w=3, depth_multiplier=opt.depth_multiplier, strides=(1, 1),
                padding='SAME', name='dwise_conv', use_bias=False,
                reuse=None):
    kernel_size = (k_w, k_h)
    in_channel = inputs.get_shape().as_list()[-1]
    filters = int(in_channel*depth_multiplier)
    return tf.layers.separable_conv2d(inputs, filters, kernel_size,
                                      strides=strides, padding=padding,
                                      data_format='channels_last', dilation_rate=(1, 1),
                                      depth_multiplier=depth_multiplier, activation=None,
                                      use_bias=use_bias, name=name, reuse=reuse
                                      )





def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid


def _fully_connected_layer(inputs, units, name="fc", activation=None, use_bias=True, reuse=None):
    return tf.layers.dense(inputs, units, activation=activation, use_bias=use_bias,
                           name=name, reuse=reuse)


def _global_avg(inputs, pool_size, strides, padding='valid', name='global_avg'):
    return tf.layers.average_pooling2d(inputs, pool_size, strides,
                                       padding=padding, data_format='channels_last', name=name)


def _squeeze_excitation_layer(input, out_dim, ratio, layer_name, is_training=True, reuse=None):
    with tf.variable_scope(layer_name, reuse=reuse):
        squeeze = _global_avg(input, pool_size=input.get_shape()[1:-1], strides=1)

        excitation = _fully_connected_layer(squeeze, units=out_dim / ratio, name=layer_name + '_excitation1',
                                            reuse=reuse)
        excitation = relu6(excitation)
        excitation = _fully_connected_layer(excitation, units=out_dim, name=layer_name + '_excitation2', reuse=reuse)
        excitation = hard_sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input * excitation
        return scale


def mobilenet_v3_block(input, k_s, expansion_ratio, output_dim, stride, name, is_training=True,
                       use_bias=True, shortcut=True, activatation="RE", ratio=16, se=False,
                       reuse=None):
    bottleneck_dim = expansion_ratio

    with tf.variable_scope(name, reuse=reuse):
        # pw mobilenetV2
        net = _conv_1x1_bn(input, bottleneck_dim, name="pw", use_bias=use_bias)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # dw
        net = _dwise_conv(net, k_w=k_s, k_h=k_s, strides=[stride, stride], name='dw',
                          use_bias=use_bias, reuse=reuse)

        net = _batch_normalization_layer(net, momentum=0.997, epsilon=1e-3,
                                         is_training=is_training, name='dw_bn', reuse=reuse)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # squeeze and excitation
        if se:
            channel = net.get_shape().as_list()[-1]
            net = _squeeze_excitation_layer(net, out_dim=channel, ratio=ratio, layer_name='se_block')

        # pw & linear
        net = _conv_1x1_bn(net, output_dim, name="pw_linear", use_bias=use_bias)

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            net += input
            net = tf.identity(net, name='block_output')

    return net
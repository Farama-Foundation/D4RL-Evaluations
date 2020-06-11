import tensorflow as tf

def build_fc_net(input_tfs, layers,
                  activation=tf.nn.relu,
                  weight_init=tf.contrib.layers.xavier_initializer(),
                  reuse=False):
    curr_tf = tf.concat(axis=-1, values=input_tfs)       
    for i, size in enumerate(layers):
        with tf.variable_scope(str(i), reuse=reuse):
            curr_tf = tf.layers.dense(inputs=curr_tf,
                                    units=size,
                                    kernel_initializer=weight_init,
                                    activation=activation)
    return curr_tf


def build_conv_net(input_tfs, layers,
                  activation=tf.nn.relu,
                  weight_init=tf.contrib.layers.xavier_initializer(),
                  reuse=False):

    img = input_tfs[0]
    img = tf.reshape(img, (-1, 48, 48, 3))
    img = tf.layers.Conv2D(filters=5, kernel_size=(5,5))(img)
    img = tf.layers.max_pooling2d(img, 2, 1)
    img = tf.layers.Conv2D(filters=5, kernel_size=(5,5))(img)
    img = tf.layers.max_pooling2d(img, 2, 1)
    img = tf.reshape(img, (-1, 5*5*9))
    # assume 1st argument is state
    if len(input_tfs) == 1:
        curr_tf = input_tfs[0]
    elif len(input_tfs) == 2:
        input_tfs[0] = img
        curr_tf = tf.concat(axis=-1, values=input_tfs)
    else:
        raise ValueError()

    for i, size in enumerate(layers):
        with tf.variable_scope(str(i), reuse=reuse):
            curr_tf = tf.layers.dense(inputs=curr_tf,
                                    units=size,
                                    kernel_initializer=weight_init,
                                    activation=activation)
    return curr_tf

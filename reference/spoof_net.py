from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Layer
from keras import backend as K
from keras.engine import InputSpec
#from keras.engine.topology import Layer
from keras.layers import activations, initializers, regularizers, constraints, Lambda
import tensorflow as tf
import numpy as np
#from keras.models import Model
#from keras.utils.generic_utils import CustomObjectScope


class ASoftmax(Dense):
    def __init__(self, units, m, batch_size,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.m = m
        self.batch_size = batch_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs):
        inputs.set_shape([self.batch_size, inputs.shape[-1]])
        inputs_norm = K.sqrt(K.sum(K.square(inputs), axis=-1, keepdims=True))
        kernel_norm = tf.nn.l2_normalize(self.kernel, dim=(0, 1))                          # W归一化
        inner_product = K.dot(inputs, kernel_norm)
        dis_cosin = inner_product / inputs_norm

        m_cosin = multipul_cos(dis_cosin, self.m)
        sum_y = K.sum(K.exp(inputs_norm * dis_cosin), axis=-1, keepdims=True)
        k = get_k(dis_cosin, self.units, self.batch_size)
        psi = np.power(-1, k) * m_cosin - 2 * k
        e_x = K.exp(inputs_norm * dis_cosin)
        e_y = K.exp(inputs_norm * psi)
        sum_x = K.sum(e_x, axis=-1, keepdims=True)
        temp = e_y - e_x
        temp = temp + sum_x

        output = e_y / temp
        return output


def multipul_cos(x, m):
    if m == 2:
        x = 2 * K.pow(x, 2) - 1
    elif m == 3:
        x = 4 * K.pow(x, 3) - 3 * x
    elif m == 4:
        x = 8 * K.pow(x, 4) - 8 * K.pow(x, 2) + 1
    else:
        raise ValueError("To high m")
    return x


def get_k(m_cosin, out_num, batch_num):
    theta_yi = tf.acos(m_cosin)  #[0,pi]
    theta_yi = tf.reshape(theta_yi, [-1])
    pi = K.constant(3.1415926)

    def cond(p1, p2, k_temp, theta):
        return K.greater_equal(theta, p2)

    def body(p1, p2, k_temp, theta):
        k_temp += 1
        p1 = k_temp * pi / out_num
        p2 = (k_temp + 1) * pi / out_num
        return p1, p2, k_temp, theta

    k_list = []
    for i in range(batch_num * out_num):
        k_temp = K.constant(0)
        p1 = k_temp * pi / out_num
        p2 = (k_temp + 1) * pi / out_num
        _, _, k_temp, _ = tf.while_loop(cond, body, [p1, p2, k_temp, theta_yi[i]])
        k_list.append(k_temp)
    k = K.stack(k_list)
    k = tf.squeeze(K.reshape(k, [batch_num, out_num]))
    return k


def asoftmax_loss(y_true, y_pred):
    d1 = K.sum(tf.multiply(y_true, y_pred), axis=-1)
    p = -K.log(d1)
    loss = K.mean(p)
    K.print_tensor(loss)
    return p


class ArcLoss(Layer):
    def __init__(self,
    			 inputs = [],
    			 name = "arc_loss"
    			 scale = 64.0,
    			 margin = 5.0,
    			 initializers = None,
                 **kwargs):
    	assert len(inputs) == 2
        super(Layer, self).__init__(**kwargs)
        self.embedding = inputs[0]
        self.label = inputs[1]
        self.name = name
        self.scale = scale
        self.margin = margin
        self.initializer = initializers

        print("  [TL] ArcLoss %s: size:%s fn:%s" % (
            self.name, self.embedding.outputs.get_shape(), self.label.outputs.get_shape()))


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs):
        inputs.set_shape([self.batch_size, inputs.shape[-1]])
        inputs_norm = K.sqrt(K.sum(K.square(inputs), axis=-1, keepdims=True))
        kernel_norm = tf.nn.l2_normalize(self.kernel, dim=(0, 1))                          # W归一化
        inner_product = K.dot(inputs, kernel_norm)
        dis_cosin = inner_product / inputs_norm

        m_cosin = multipul_cos(dis_cosin, self.m)
        sum_y = K.sum(K.exp(inputs_norm * dis_cosin), axis=-1, keepdims=True)
        k = get_k(dis_cosin, self.units, self.batch_size)
        psi = np.power(-1, k) * m_cosin - 2 * k
        e_x = K.exp(inputs_norm * dis_cosin)
        e_y = K.exp(inputs_norm * psi)
        sum_x = K.sum(e_x, axis=-1, keepdims=True)
        temp = e_y - e_x
        temp = temp + sum_x

        output = e_y / temp
        return output

    def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.5):
        '''
        :param embedding: the input embedding vectors
        :param labels:  the input labels, the shape should be eg: (batch_size, 1)
        :param s: scalar value default is 64
        :param out_num: output class num
        :param m: the margin value, default is 0.5
        :return: the final cacualted output, this output is send into the tf.nn.softmax directly
        '''
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = sin_m * m  # issue 1
        threshold = math.cos(math.pi - m)
        with tf.variable_scope('arcface_loss'):
            # inputs and weights norm
            embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
            embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
            weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                      initializer=w_init, dtype=tf.float32)
            weights_norm = tf.norm(weights, axis=0, keepdims=True)
            weights = tf.div(weights, weights_norm, name='norm_weights')
            # cos(theta+m)
            cos_t = tf.matmul(embedding, weights, name='cos_t')
            cos_t2 = tf.square(cos_t, name='cos_2')
            sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
            sin_t = tf.sqrt(sin_t2, name='sin_t')
            cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

            # this condition controls the theta+m should in range [0, pi]
            #      0<=theta+m<=pi
            #     -m<=theta<=pi-m
            cond_v = cos_t - threshold
            cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

            keep_val = s*(cos_t - mm)
            cos_mt_temp = tf.where(cond, cos_mt, keep_val)

            mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
            # mask = tf.squeeze(mask, 1)
            inv_mask = tf.subtract(1., mask, name='inverse_mask')

            s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
            
            logit = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
            inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
            
    #        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
        return inference_loss,logit

class SpoofDenseNet:
	@staticmethod
	def build(feature_size, classes, reg, init="glorot_uniform"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()

		model.add(Dense(256, kernel_initializer=init, kernel_regularizer=reg, input_shape=(feature_size,)))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.75))

		model.add(Dense(256, kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.75))

		model.add(Dense(128, kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.75))

		model.add(Dense(64, kernel_initializer=init, kernel_regularizer=reg))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.75))
 
		# softmax classifier
		model.add(Dense(classes))
		model.add(BatchNormalization())
		#model.add(Activation("softmax"))
		model.add(ASoftmax(3,3,32))
 
		# return the constructed network architecture
		return model
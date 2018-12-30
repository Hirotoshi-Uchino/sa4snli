from keras import backend as K
from keras.engine.topology import Layer

# Multi-Head Self-Attention Layer
class MHSA(Layer):
    def __init__(self, heads=1, use_softmax=True, **kwargs):
        self.heads       = heads
        self.use_softmax = use_softmax
        super(MHSA, self).__init__(**kwargs)

    def build(self, input_shape):

        self.wq_kernels = []
        self.wk_kernels = []
        self.wv_kernels = []
        head_dim = int(input_shape[2] / self.heads)

        for i in range(self.heads):
            self.wq_kernels.append(self.add_weight(name='kernel',
                                      shape=(input_shape[2], head_dim),
                                      initializer='glorot_uniform',
                                      trainable=True))
            self.wk_kernels.append(self.add_weight(name='kernel',
                                      shape=(input_shape[2], head_dim),
                                      initializer='glorot_uniform',
                                      trainable=True))
            self.wv_kernels.append(self.add_weight(name='kernel',
                                      shape=(input_shape[2], head_dim),
                                      initializer='glorot_uniform',
                                      trainable=True))

        self.wo_kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2], input_shape[2]),
                                      initializer='glorot_uniform',
                                      trainable=True)

        super(MHSA, self).build(input_shape)

    def call(self, x):
        attentions = self.__compute_scaled_dot_product_attentions(x)
        return K.dot(K.concatenate(attentions, axis=2), self.wo_kernel)


    def compute_output_shape(self, input_shape):
        return (input_shape)


    def __compute_scaled_dot_product_attentions(self, x):
        attentions = []
        sqdk = K.get_variable_shape(x)[2] / self.heads

        for i in range(self.heads):
            qw = K.dot(x, self.wq_kernels[i])
            kw = K.dot(x, self.wk_kernels[i])
            vw = K.dot(x, self.wv_kernels[i])

            scaled_qwkw = K.batch_dot(qw, kw, axes=[2, 2]) / sqdk
            if self.use_softmax:
                scaled_qwkw = K.softmax(scaled_qwkw)

            attention = K.batch_dot(scaled_qwkw, vw, axes=[2,1])
            attentions.append(attention)

        return attentions

# Position-wise Feed-Forward Networks
class PFNN(Layer):

    def __init__(self, ff_dim, **kwargs):
        self.ff_dim = ff_dim
        super(PFNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel_1 = self.add_weight(name='kernel',
                                      shape=(input_shape[2], self.ff_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernel_2 = self.add_weight(name='kernel',
                                        shape=(self.ff_dim, input_shape[2]),
                                        initializer='glorot_uniform',
                                        trainable=True)
        self.bias_1   = self.add_weight(name='bias', shape=(1, self.ff_dim),
                                      initializer='zeros', trainable=True)
        self.bias_2   = self.add_weight(name='bias', shape=(1, input_shape[2]),
                                        initializer='zeros', trainable=True)
        super(PFNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        hidden = K.dot(x, self.kernel_1) + self.bias_1
        hidden = K.relu(hidden)

        out = K.dot(hidden, self.kernel_2) + self.bias_2
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape)
from SNLI import SNLI
from layers import MHSA, PFNN

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization



snli = SNLI()

# generate data
#snli.generate_data('../data/pickle/snli_concat_glove6B.300d.pickle')

# load data
snli = snli.load_data('../data/pickle/snli_pad_concat_glove6B.300d.pickle')

num_words     = snli.embedding_matrix.shape[0]
embedding_dim = 300
dim           = 200
class_dim     = 3
batch_size    = 512
epochs        = 5
keep_prob     = 1.
use_PE        = False
use_softmax   = False

# ==================================
# construct model
# ==================================


input = Input(shape=(snli.x_train[0].shape))
x     = Embedding(input_dim=num_words, output_dim=embedding_dim,
                  weights=[snli.embedding_matrix], trainable=True)(input)

# TODO: add positional encoding
if use_PE:
    import numpy as np
    from nlp_util import create_positional_encoding
    from keras import backend as K

    pe1 = create_positional_encoding(length=snli.s1_maxlen, dim_model=embedding_dim, wlength=50.)
    pe2 = create_positional_encoding(length=snli.s2_maxlen, dim_model=embedding_dim, wlength=50.)
    sep_pe = np.zeros((1, embedding_dim))
    pe  = np.concatenate([pe1, sep_pe, pe2])
    pe = K.constant(pe)
    x  = Lambda(lambda x: x + pe)(x)

x = MHSA(heads=2, use_softmax=use_softmax)(x)
x = PFNN(300)(x)
#x = Dropout(keep_prob)(x)

# x = MHSA(heads=4, use_softmax=use_softmax)(x)
# x = PFNN(300)(x)
#x = Dropout(keep_prob)(x)


x     = Flatten()(x)

# x    = Dense(1000)(x)
# x    = PReLU()(x)
# x    = BatchNormalization()(x)
# #
# x    = Dense(100)(x)
# x    = PReLU()(x)
# x    = BatchNormalization()(x)
#x    = Dropout(keep_prob)(x)
#x   = Dense(dim)(x)
#x   = PReLU()(x)

out   = Dense(class_dim, activation='softmax')(x)


model = Model(inputs=input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(snli.x_train, snli.y_train, batch_size=batch_size, epochs=epochs,
        validation_data=(snli.x_dev, snli.y_dev), shuffle=True)

print(model.summary())

score, acc = model.evaluate(snli.x_test, snli.y_test, batch_size=batch_size)

print("Test score:", score)
print("Test accuracy:", acc)


save_model = True
if save_model:
    from keras.utils.vis_utils import plot_model
    model.save('../model/mhsa_model_01.h5')
    plot_model(model, to_file='../model/mhsa_model_02.svg', show_shapes=True)

from SNLI import SNLI
from layers import MHSA, PFNN

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization



snli = SNLI()

# generate data
#snli.generate_data('../data/pickle/snli_concat_glove6B.300d.pickle')

# load data
snli = snli.load_data('../data/pickle/snli_concat_glove6B.300d.pickle')

num_words     = snli.embedding_matrix.shape[0]
embedding_dim = 300
dim           = 200
class_dim     = 3
batch_size    = 512
epochs        = 5
keep_prob     = 0.5


# ==================================
# construct model
# ==================================


input = Input(shape=(snli.x_train[0].shape))
x     = Embedding(input_dim=num_words, output_dim=embedding_dim,
                  weights=[snli.embedding_matrix], trainable=True)(input)

x     = MHSA(heads=2, use_softmax=True)(x)
x     = PFNN(300)(x)

x     = MHSA(heads=2, use_softmax=True)(x)
x     = PFNN(300)(x)

x     = Flatten()(x)



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


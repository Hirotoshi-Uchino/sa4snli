

from ..SNLI import SNLI

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Dropout
from keras.layers.advanced_activations import PReLU

snli = SNLI()
# load data
snli = snli.load_data('../../data/pickle/snli_concat_glove6B.300d.pickle')


num_words     = snli.embedding_matrix.shape[0]
embedding_dim = 300
dim           = 200
class_dim     = 3
batch_size    = 512

input = Input(shape=(snli.x_train[0].shape))
x     = Embedding(input_dim=num_words, output_dim=embedding_dim,
                  weights=snli.embedding_matrix, trainable=True)(input)
x     = Dense(dim)(x)
x     = PReLU()(x)

x     = Dense(dim)(x)
x     = PReLU()(x)

out   = Dense(class_dim, activation='sigmoid')(x)

model = Model(inputs=input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(snli.x_train, snli.y_train, batch_size=batch_size,
          validation_data=[snli.x_train, snli.y_train], shuffle=True)

print(model.summary())

score, acc = model.evaluate(snli.x_test, snli.y_test, batch_size=batch_size)

print("Test score:", score)
print("Test accuracy:", acc)
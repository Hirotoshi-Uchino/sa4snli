import numpy as np
from keras.models import load_model

import sys
sys.path.append('../sa4snli/')

from SNLI import SNLI
from layers import MHSA, PFNN

snli = SNLI()

batch_size    = 512
epochs        = 1

# load data
snli = snli.load_data('../data/pickle/snli_pad_concat_glove6B.300d.pickle')
model = load_model('../model/mhsa_model_02.h5', custom_objects={'MHSA': MHSA, 'PFNN': PFNN})

index_to_word = {i: w for w, i in snli.tokenizer.word_index.items()}
index_to_word[0] = ''

index_to_label = {0: "contradiction", 1: "entailment", 2: "neutral"}

predict = model.predict(snli.x_test).argmax(axis=1)
correct = snli.y_test.argmax(axis=1)

i2w = lambda i: index_to_word[i]


path_m = '../data/test_inferred/match_results_02.tsv'
path_u = '../data/test_inferred/unmatch_results_02.tsv'

file_m = open(path_m, mode='w')
file_u = open(path_u, mode='w')

for i, (p, c, sentence) in enumerate(zip(predict, correct, snli.x_test)):
    sl = np.vectorize(i2w)(sentence)
    sentence = ' '.join(sl[~(sl == '')])

    if p == c:
        line = str(i) + '\t' + sentence + '\t' + index_to_label[p] + '\t' + index_to_label[c] + '\n'
        file_m.write(line)
        # print(sentence, index_to_label[p], index_to_label[c])
    else:
        line = str(i) + '\t' + sentence + '\t' + index_to_label[p] + '\t' + index_to_label[c] + '\n'
        file_u.write(line)
        # print(sentence, index_to_label[p], index_to_label[c])


file_m.close()
file_u.close()

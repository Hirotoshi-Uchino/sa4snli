from keras.models import Sequential


from SNLI import SNLI



snli = SNLI()

# generate data
#snli.generate_data('../data/pickle/snli_concat_glove6B.300d.pickle')

# load data
snli = snli.load_data('../data/pickle/snli_concat_glove6B.300d.pickle')


# ==================================
# construct model
# ==================================

#

# ==================================
#
# ==================================


example of config.ini file
======================
::
   [data_path]
   BASE_DIR=../NLP
   WORD_EMBED=%(BASE_DIR)s/word_embed/glove.6B.300d.txt
   SNLI=%(BASE_DIR)s/SNLI/snli_1.0
   TRAIN_DATA=%(SNLI)s/snli_1.0_train.txt
   DEV_DATA=%(SNLI)s/snli_1.0_dev.txt
   TEST_DATA=%(SNLI)s/snli_1.0_test.txt


usage of SNLI class
======================
.. code-block:: python
   from SNLI import SNLI
   snli = SNLI()

   # generate data
   snli.generate_data('../data/snli_concat_glove6B.300d.pickle')

   # load data
   snli = snli.load_data('../data/snli_concat_glove6B.300d.pickle')



# Self-Attentive Model

Implementation of A Structured Self-attentive Sentence Embedding [[Paper](https://arxiv.org/abs/1703.03130)] [[Review](https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/self_attention_embedding.md)]

To run the model, [GloVe](https://nlp.stanford.edu/projects/glove/) word vectors need to be downloaded and placed in 'word2vecs'. By default, we use 200d vectors trained 
on Wikipedia corpus.

``` shell
mkdir word2vecs
```

'Data Preprocessing.ipynb' helps preprocess raw reviews into format compatible with data in 'data' folder.

Training, validation and test data ending with '.csv' file is put in 'data' folder. The first column is review token sequence, while the second column corresponds to 2 classes (helpful or not). The last column is the token length and the review is sorted in descending order. 

'data/embed_matrix.pt' is the word embedding matrix initialized with GloVe vectors and words do not show up in GloVe is initialed from Uniform distribution. The corresponding index in stored in 'data/word_idx_list.pt'. This can help only update word embeddings that do not appear in GloVe.

To start training, run
``` shell
python train.py
```

After training, we can get the attention weights of test data by
``` shell
python attentive_weights.py
```

Then we can turn to 'Attention Visualization.ipynb' to highlight word tokens in reviews.

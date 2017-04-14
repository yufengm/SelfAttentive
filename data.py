import os
import torch
import csv
from nltk.tokenize import word_tokenize, sent_tokenize

# Data Class
class Dictionary( object ):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append( word )
            self.word2idx[ word ] = len(self.idx2word) - 1
        return self.word2idx[ word ]

    def __len__(self):
        return len( self.idx2word )


class Corpus( object ):
    def __init__(self, path):
        self.dictionary = Dictionary()
        
        # Load all train, valid and test data
        self.train, self.train_len, self.train_label = self.tokenize( os.path.join( path, 'train.csv' ) )
        self.valid, self.valid_len, self.valid_label = self.tokenize( os.path.join( path, 'valid.csv' ) )
        self.test, self.test_len, self.test_label = self.tokenize( os.path.join( path, 'test.csv' ) )

    def tokenize( self, path ):
        """Tokenizes a csv file."""
        assert os.path.exists( path )
        # Add words to the dictionary
        max_len = 0
        with open( path, 'r' ) as f:
            Reader = csv.reader( f, delimiter=',', quoting=csv.QUOTE_MINIMAL )
            count = 0
            for record in Reader:
                count += 1
                review = record[ 0 ].decode( 'utf-8' )
                # words = [ word for sent in sent_tokenize( review ) for word in word_tokenize( sent ) ]
                words = review.split()
                if len( words ) > max_len:
                    max_len = len( words )
                for word in words:
                    self.dictionary.add_word( word )

        # Tokenize file content
        with open( path, 'r' ) as f:
            Reader = csv.reader( f, delimiter=',', quoting=csv.QUOTE_MINIMAL )
            data = torch.LongTensor( count, max_len ).fill_( 0 )
            lengths = []
            labels = torch.LongTensor( count ).fill_( 0 )

            for idx, record in enumerate( Reader ):
                labels[ idx ] = int( record[ 1 ] )
                review = record[ 0 ].decode( 'utf-8' )
                # words = [ word for sent in sent_tokenize( review ) for word in word_tokenize( sent ) ]
                words = review.split()
                lengths.append( len( words ) )
                
                for i, word in enumerate( words ):
                    data[ idx, i ] = self.dictionary.word2idx[word]
                    
        return data, lengths, labels
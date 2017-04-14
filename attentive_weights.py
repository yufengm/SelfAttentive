import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from data import *
from model import *
from Utility import *

def evaluate( data_source, labels, data_len ):
    total_loss = 0
    
    acc = []
    pre = []
    rec = []
    f1 = []
    
    ntokens = len( corpus.dictionary )
    hidden = model.init_hidden( test_batch_size )
    nclass = 2
    
    for i in range( 0, data_source.size(0) - 1, test_batch_size ):
        
        data, targets, len_li = get_batch( data_source, labels, data_len, i, test_batch_size, cuda, evaluation=True )
        output, hidden, penal, weights = model( data, hidden, len_li )
        output_flat = output.view( -1, nclass )
        
        total_loss += data.size( 0 ) * ( entropy_loss( output_flat, targets ).data + penal.data )
        hidden = repackage_hidden( hidden )
        
        _, pred = output_flat.topk( 1 , 1, True, True )
        pred = pred.t()
        target = targets.view( 1, -1 )
        
        p, r, f, a = compute_measure( pred, target )
        acc.append( a )
        pre.append( p )
        rec.append( r )
        f1.append( f )
    
    # Compute Precision, Recall, F1, and Accuracy
    print 'Measure on this dataset'
    print 'Precision:', np.mean( pre )
    print 'Recall:', np.mean( rec )
    print 'F1:', np.mean( f1 )
    print 'Acc:', np.mean( acc )

    return total_loss[0] / len( data_source )

cuda = True
path = 'data/'
test_batch_size = 32

# Define the Loss Function for Training
entropy_loss = nn.CrossEntropyLoss()
if cuda:
    entropy_loss.cuda()

# Loading Test Data
corpus = Corpus( path )
print 'Dataset loaded.'
    
# Make test data batchifiable
test_data = select_data( corpus.test, test_batch_size )
test_len = select_data( corpus.test_len, test_batch_size )
test_label = select_data( corpus.test_label, test_batch_size )
    
# Load Trained Model
with open( 'Attentive-best.pt', 'rb' ) as f:
    model = torch.load( f )
print 'Model loaded.'
    
# Calculate and Save weights
Weights = Attentive_weights( model, test_data, test_label, test_len, test_batch_size, cuda )

torch.save( Weights, 'attention_weights.pt' )

test_loss = evaluate( test_data, test_label, test_len )

print( '=' * 80 )
print( '| End of training | test loss {:5.2f} |'.format( test_loss ) )
print( '=' * 80 )

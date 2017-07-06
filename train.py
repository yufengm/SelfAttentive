import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import csv
import numpy as np

# Import Model and Data from external file
from model import *
from data import *
from Utility import *


##########################################################################
#                         CommandLine Argument Setup                     #
##########################################################################

parser = argparse.ArgumentParser(description='PyTorch Self-Attentive Sentence Embedding Model')
parser.add_argument( '-f', default='self', help='To make it runnable in jupyter' )
parser.add_argument('--data', type=str, default='./data/', help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--r', type=int, default=30, help='r in paper, # of keywords you want to focus on')
parser.add_argument('--mlp_nhid', type=int, default=3000, help='r in paper, # of keywords you want to focus on')
parser.add_argument('--da', type=int, default=350, help='da in paper' )
parser.add_argument('--lamb', type=float, default=1, help='penalization term coefficient')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='sgd with momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--eval_size', type=int, default=32, metavar='N', help='evaluation batch size')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--pretrained', type=str, default='', help='whether start from pretrained model')
parser.add_argument('--cuda', default=False, action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='report interval')
parser.add_argument('--save', type=str,  default='Attentive-best.pt', help='path to save the final model')

args = parser.parse_args()

# # Define the corpus and load data
corpus = Corpus( args.data )
print('Training data loaded.....')

# Make Dataset batchifiable
train_data = select_data( corpus.train, args.batch_size )
train_len = select_data( corpus.train_len, args.batch_size )
train_label = select_data( corpus.train_label, args.batch_size )

val_data = select_data( corpus.valid, args.eval_size )
val_len = select_data( corpus.valid_len, args.eval_size )
val_label = select_data( corpus.valid_label, args.eval_size )

test_data = select_data( corpus.test, args.eval_size )
test_len = select_data( corpus.test_len, args.eval_size )
test_label = select_data( corpus.test_label, args.eval_size )


# Pretrained Glove word vectors and Word Indices in Embedding matrix that are not included in Glove
if args.cuda:
    emb_matrix = torch.load( args.data + 'emb_matrix.pt' )
    word_idx_list = torch.load( args.data + 'word_idx_list.pt' )
    emb_matrix.cuda()
    word_idx_list.cuda()
else:
    emb_matrix = torch.load( args.data + 'emb_matrix.pt', map_location=lambda storage, loc: storage )
    word_idx_list = torch.load( args.data + 'word_idx_list.pt', map_location=lambda storage, loc: storage )


# Define Model
ntokens = len( corpus.dictionary )
nclass = 2

if not args.pretrained:
    model = SelfAttentive( ntokens, args.emsize, args.nhid, args.nlayers, args.da, args.r, args.mlp_nhid, nclass, emb_matrix, args.cuda )
else:
    model = torch.load( args.pretrained )
    print('Pretrained model loaded.')

entropy_loss = nn.CrossEntropyLoss()
if args.cuda:
    model.cuda()
    entropy_loss.cuda()


# Define the training function
def train( lr, epoch ):
    # word_update: whether glove vectors are updated

    total_loss = 0
    start_time = time.time()
    all_losses = []

    hidden = model.init_hidden( args.batch_size )

    # Per-parameter training
    params = list( model.parameters() )

    optimizer = torch.optim.SGD( params, lr, momentum = args.momentum, weight_decay = args.weight_decay )

    for batch_idx, start_idx in enumerate( range( 0, train_data.size(0) - 1, args.batch_size ) ):

        # Retrieve one batch for training
        data, targets, len_li = get_batch( train_data, train_label, train_len, start_idx, args.batch_size, args.cuda )
        hidden = repackage_hidden( hidden )

        output, hidden, penal, weights = model( data, hidden, len_li )

        # Loss = cross_entropy + lambda * penalization
        loss = entropy_loss( output.view( -1, nclass ), targets ) + args.lamb * penal

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping in case of gradient explosion
        torch.nn.utils.clip_grad_norm( model.parameters(), args.clip )

        optimizer.step()

        total_loss += loss.data
        all_losses.append( loss.data )


        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} |'.format(
                epoch, batch_idx, len( train_data ) // args.batch_size, lr,
                elapsed * 1000 / args.log_interval, cur_loss ) )

            total_loss = 0
            start_time = time.time()

    return np.mean( all_losses )


# Define evaluation function
def evaluate( data_source, labels, data_len ):
    total_loss = 0

    acc = []
    pre = []
    rec = []
    f1 = []

    ntokens = len( corpus.dictionary )
    hidden = model.init_hidden( args.eval_size )

    for i in range( 0, data_source.size(0) - 1, args.eval_size ):

        data, targets, len_li = get_batch( data_source, labels, data_len, i, args.eval_size, args.cuda, evaluation=True )
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
    print('Measure on this dataset')
    print('Precision:', np.mean( pre ))
    print('Recall:', np.mean( rec ))
    print('F1:', np.mean( f1 ))
    print('Acc:', np.mean( acc ))

    return total_loss[0] / len( data_source )


# Training Process
print('Start training...')
lr = args.lr
best_val_loss = None
all_losses = []
print('# of Epochs:', args.epochs)

for epoch in range( 1, args.epochs + 1 ):
    epoch_start_time = time.time()
    all_losses.append( train( lr, epoch )[0] )
    val_loss = evaluate( val_data, val_label, val_len )
    print( '-'*80 )
    print( '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
          .format( epoch, (time.time() - epoch_start_time), val_loss ) )
    print( '-'*80 )

    # Save the best model and Anneal the learning rate.
    if not best_val_loss or val_loss < best_val_loss:
        best_val_loss = val_loss
        with open( args.save , 'wb' ) as f:
            torch.save( model , f )
    else:
        lr /= 4.0


# Plot the Learning Curve
plt.figure()
plt.plot( all_losses )
plt.savefig( 'Learning-curve.eps' )


# Run on test data and Save the model.
with open( args.save, 'rb' ) as f:
    model = torch.load( f )

test_loss = evaluate( test_data, test_label, test_len )

print( '=' * 80 )
print( '| End of training | test loss {:5.2f} |'.format( test_loss ) )
print( '=' * 80 )

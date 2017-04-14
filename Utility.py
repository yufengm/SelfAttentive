import torch
from torch.autograd import Variable

# Batchify the whole dataset
def select_data( data, bsz ):
    try:
        nbatch = data.size( 0 ) // bsz
        data = data.narrow( 0, 0, nbatch * bsz )
    
        if args.cuda:
            data = data.cuda()
    except:
        nbatch = len( data ) // bsz
        data = data[ : nbatch * bsz ]
        
    return data


# Unpack the hidden state
def repackage_hidden( h ):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type( h ) == Variable:
        return Variable( h.data )
    else:
        return tuple( repackage_hidden( v ) for v in h )

# Retrieve a batch from the source
def get_batch( source, labels, len_list, i, size, cuda, evaluation=False ):
    
    batch_size = size
    data = source[ i : i + batch_size ]
    if cuda:
        data = Variable( data.cuda() , volatile = evaluation )
        target = Variable( labels[ i : i + batch_size ].view( -1 ).cuda() )
        len_li = Variable( torch.LongTensor( len_list[ i : i + batch_size ] ).cuda() )
    else:
        data = Variable( data , volatile = evaluation )
        target = Variable( labels[ i : i + batch_size ].view( -1 ) )
        len_li = Variable( torch.LongTensor( len_list[ i : i + batch_size ] ) )
    
    return data, target, len_li


# Function to compute precision, recall, f1 and accuracy
def compute_measure( pred, target ):
    pred = pred.view(-1)
    target = target.view(-1)
    
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range( pred.size(0) ):
        if pred.data[ i ] == 1 and target.data[ i ] == 1:
            tp += 1
        elif pred.data[ i ] == 1 and target.data[ i ] == 0:
            fp += 1
        elif pred.data[ i ] == 0 and target.data[ i ] == 1:
            fn += 1
        else:
            tn += 1

    pre = tp / float( fp + tp + 1e-8 )
    rec = tp / float( fn + tp + 1e-8 )
    f1 = 2 * pre * rec / ( pre + rec + 1e-8 )
    acc = ( tn + tp ) / float( tn + fp + fn + tp + 1e-8 )
            
    return pre, rec, f1, acc


# Get Attention Weights Function
def Attentive_weights( model, data_source, labels, data_len, eval_batch_size, cuda ):
    
    hidden = model.init_hidden( eval_batch_size )
    Weights = {}
    
    for i in range( 0, data_source.size( 0 ) - 1, eval_batch_size ):
        
        data, targets, len_li = get_batch( data_source, labels, data_len, i, eval_batch_size, cuda, evaluation=True )
        output, _, _, weights = model( data, hidden, len_li )
        
        Weights[ i ] = [ weights, output ]

    return Weights

#from __future__ import print_function
#from __future__ import division

import numpy as np
import sys,os
sys.path.append("cbp/")
import pdb
import pickle

import utils
from cbp import bilinear_pool
import keras.activations as activations
from keras.initializers import Constant
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import ModelCheckpointPickle
from keras.layers import Input, TimeDistributed, BatchNormalization
from keras.layers.merge import add, concatenate, multiply
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dense, Dropout, Lambda, Permute, RepeatVector, Reshape

from keras import backend as K
import tensorflow as tf

def config():
    trainf_name = 'train_data'
    valf_name = 'val_data'
    testf_name = 'val_data'
    imf_name = 'res_feats'
    pkl_name = trainf_name + '_' + valf_name + '_' + testf_name + '_' + imf_name
    c = dict()
    c['pretrained_model'] = None
    c['load_pickle'] = True
    c['pickle_name'] = 'pkl/'+pkl_name+'.pkl'
    c['pickle_load_weights'] = True
    c['trainf'] = 'csv/' + trainf_name+'.csv'
    c['valf'] = 'csv/' + valf_name+'.csv'
    c['testf'] = 'csv/' + testf_name+'.csv'

    c['nfold'] = 10
    c['val_ratio'] = 0.1

    # embedding params
    c['emb'] = 'Glove'
    c['emb_dim'] = 300
    c['pe'] = True
    c['im_dim'] = 2048
    c['cbp_dim'] = 1024

    # training hyperparams
    c['opt'] = 'adam'
    c['batch_size'] = 32
    c['epochs'] = 32
    c['patience'] = 10 
    c['margin'] = 0.1
    c['l2reg'] = 1e-5
    c['lr'] = 0.001

    # sentences with word lengths below the 'pad' will be padded with 0.
    c['pad'] = 40
    c['spad'] = 60

    # write networks
    c['mem_dim'] = 300
    c['w_conv_f_v'] = 10
    c['w_conv_s_v'] = 5
    c['w_conv_s_h'] = 1
    c['n_WN_layers'] = 1
    c['cnnact'] = 'relu'

    # read networks
    c['r_conv_f_v'] = 3
    c['r_conv_s_v'] = 1
    c['r_conv_s_h'] = 1
    c['n_RN_layers'] = 1

    ps, h = utils.hash_params(c)

    return c, ps, h

def load_data(trainf, valf, testf):
    global vocab, inp_tr, inp_val, inp_test
    
    if conf['load_pickle']:
        print ('load data from %s' % conf['pickle_name'])
        with open(conf['pickle_name'], 'rb') as h:
            ll = pickle.load(h)
            inp_tr, inp_val, inp_test, vocab = ll[0], ll[1], ll[2], ll[3]
    else:
        print ('Not supported yet')

def pe(encoder, dims):
    def func(x):
        emb = x
        if dims == 2:
            pos_val = K.constant(value=np.arange(conf['pad'])) # shape=(pad,)
            pos_val_3d = K.expand_dims(pos_val, axis=0) # shape=(1, pad)
            pos_val_3d = K.tile(pos_val_3d, (K.shape(x)[0], 1)) # shape=(batch_size_of_x, pad)

            encoding_3d = encoder(pos_val_3d) # shape=(batch_size_of_x, pad, N)
            emb = add([emb, encoding_3d])
        elif dims == 3:
            pos_val = K.constant(value=np.arange(conf['pad'])) # shape=(pad,)
            pos_val_4d = K.expand_dims(pos_val, axis=0)
            pos_val_4d = K.tile(pos_val_4d, (x._keras_shape[1], 1)) # shape=(spad, pad)
            pos_val_4d = K.expand_dims(pos_val_4d, axis=0)
            pos_val_4d = K.tile(pos_val_4d, (K.shape(x)[0], 1, 1)) # shape=(None, spad, pad)

            encoding_4d = TimeDistributed(encoder)(pos_val_4d) # shape=(None, spad, pad, N))
            emb = add([emb, encoding_4d])
        return emb

    return Lambda(func, output_shape=lambda shape:shape)

def cbp(out_dim, inp_dim1, inp_dim2):
    def func(x):
        inp1, inp2 = x[:,:inp_dim1], x[:,inp_dim1:]
        return bilinear_pool(inp1, inp2, out_dim)
    return Lambda(func, output_shape=lambda shape:(None,) + (out_dim,))

def embedding(pfx='emb'):
    spad = conf['spad']
    pad = conf['pad']
    im_dim = conf['im_dim']
    
    input_qi = Input(name='qi', batch_shape=(None, pad), dtype='int32')                          
    input_si = Input(name='si', batch_shape=(None, spad, pad), dtype='int32')                 
    input_im = Input(name='im', batch_shape=(None, spad, im_dim))

    input_a_0i = Input(name='a_0i', batch_shape=(None, pad), dtype='int32')                        
    input_a_1i = Input(name='a_1i', batch_shape=(None, pad), dtype='int32')         
    input_a_2i = Input(name='a_2i', batch_shape=(None, pad), dtype='int32')                        
    input_a_3i = Input(name='a_3i', batch_shape=(None, pad), dtype='int32')         
    input_a_4i = Input(name='a_4i', batch_shape=(None, pad), dtype='int32')         

    
    input_nodes = [input_qi, input_si, input_a_0i, input_a_1i, input_a_2i, input_a_3i, input_a_4i, input_im] 

    shared_embedding1 = Embedding(name='Glove1_'+pfx, input_dim=vocab.size(), input_length=spad*pad,
                                output_dim=emb.N, mask_zero=False, batch_input_shape=(None, spad*pad),
                                embeddings_initializer=Constant(vocab.embmatrix(emb)), trainable=False)
    shared_embedding2 = Embedding(name='Glove2_'+pfx, input_dim=vocab.size(), input_length=pad,
                                output_dim=emb.N, mask_zero=False, batch_input_shape=(None, pad),
                                embeddings_initializer=Constant(vocab.embmatrix(emb)), trainable=False)

    N = emb.N
    emb_ai = []
    emb_si = Reshape((spad*pad,), input_shape=(spad, pad))(input_si)
    emb_si = shared_embedding1(input_si)
    emb_si = Reshape((spad, pad, emb.N))(emb_si)

    emb_qi = shared_embedding2(input_qi)
    emb_ai.append(shared_embedding2(input_a_0i))
    emb_ai.append(shared_embedding2(input_a_1i))
    emb_ai.append(shared_embedding2(input_a_2i))
    emb_ai.append(shared_embedding2(input_a_3i))
    emb_ai.append(shared_embedding2(input_a_4i))

    if conf['pe']:
        encoder = Embedding(name='pe_learnable_layer', input_dim=conf['pad'], input_length=conf['pad'], input_shape=(conf['pad'],),
                                        output_dim=N, mask_zero=False, trainable=True)
        emb_qi = pe(encoder, 2)(emb_qi)
        emb_si = pe(encoder, 3)(emb_si)
        for i in range(5):
            emb_ai[i] = pe(encoder, 2)(emb_ai[i])

    avg_layer_3d = Lambda(name='avg3d_'+pfx, function=lambda x: K.mean(x, axis=2), output_shape=lambda shape:(shape[:2]) + shape[3:])
    emb_si = avg_layer_3d(emb_si) # shape=(None, spad, N)

    avg_layer_2d = Lambda(name='avg2d_'+pfx, function=lambda x: K.mean(x, axis=1), output_shape=lambda shape:(shape[:1]) + shape[2:])
    emb_qi = avg_layer_2d(emb_qi) # shape=(None, N)
    for i in range(5):
        emb_ai[i] = avg_layer_2d(emb_ai[i]) # shape=(None, N)

    # cbp fusion between image and text
    emb_im_si = TimeDistributed(cbp(conf['cbp_dim'], conf['emb_dim'], conf['im_dim']))(concatenate([emb_si, input_im])) # shape=(None, spad, cbp_dim)

    fc = Dense(conf['mem_dim'], kernel_regularizer=l2(conf['l2reg']), activation='linear', name='FC_q_'+pfx)
    u = fc(emb_qi) # shape = (None, mem_dim)
    for i in range(5):
        emb_ai[i] = fc(emb_ai[i]) # shape = (None, mem_dim)

    return input_nodes, u, emb_im_si, emb_ai

def write_networks(E, pfx='WN'):
    # input shape=(None, spad, cbp_dim)
    # output shape=(None, m, mem_dim, 3)

    fc = Dense(conf['mem_dim'], kernel_regularizer=l2(conf['l2reg']), activation='linear', name='FC_'+pfx)
    fc_out = TimeDistributed(fc, name='TD_fc_'+pfx)(E) # shape = (None, spad, mem_dim)

    expand_dim = Lambda(name='expand_dim_'+pfx, function=lambda x: K.expand_dims(x, axis=-1), output_shape=lambda shape:shape+(1,))
    conv_in = expand_dim(fc_out) # shape = (None, spad, mem_dim, 1)
    for l in range(conf['n_WN_layers']):
        conv = Convolution2D(3, kernel_size=(conf['w_conv_f_v'], conf['mem_dim']), strides=(conf['w_conv_s_v'], conf['w_conv_s_h']), padding='same',
                        name='w_conv%d'%(l), activation='linear', kernel_regularizer=l2(conf['l2reg']))
        conv_out = conv(conv_in) # shape = (None, m, mem_dim, 3)
        conv_out = Activation(conf['cnnact'])(BatchNormalization()(conv_out))
        conv_in = conv_out

    return conv_out

def q_tile(m, pfx):
    def func(x):
       x = K.expand_dims(x, axis=1) # shape=(None, 1, mem_dim) 
       x = K.tile(x, (1,m,1)) # shape=(None, m, mem_dim)
       x = K.expand_dims(x, axis=3) # shape=(None, m, mem_dim, 1)
       x = K.tile(x, (1,1,1,3)) # shape=(None, m, mem_dim, 3)
       return x
    return Lambda(func, output_shape=lambda shape:(shape[0],)+(m,)+(conf['mem_dim'],)+(3,), name='q_tile_'+pfx)

def cbp_M_q(m):
    def func(x):
        M = x[0] # shape=(None, m, mem_dim, 3)
        u = x[1] # shape=(None, m, mem_dim, 3)

        M = K.permute_dimensions(M, (0, 1, 3, 2)) # shape=(None, m, 3, mem_dim)
        M = Reshape((m*3, conf['mem_dim']))(M) # shape=(None, m*3, mem_dim)
        
        u = K.permute_dimensions(u, (0, 1, 3, 2)) # shape=(None, m, 3, mem_dim)
        u = Reshape((m*3, conf['mem_dim']))(u) # shape=(None, m*3, mem_dim)

        M_q = TimeDistributed(cbp(conf['mem_dim'], conf['mem_dim'], conf['mem_dim']))(concatenate([M, u])) # shape=(None, m*3, mem_dim)
        M_q = Reshape((m, 3, conf['mem_dim']))(M_q) # shape = (None, m, 3, mem_dim)
        M_q = K.permute_dimensions(M_q, (0, 1, 3, 2)) # shape = (None, m, mem_dim, 3)

        return M_q
    return Lambda(func, output_shape=lambda shape:shape)

def read_networks(M, u, pfx='RN'):
    # input shape
    # M: (None, m, mem_dim, 3)
    # q: (None, mem_dim)
    # output shape=(None, c, 3)
 
    m = M._keras_shape[1]
    u_ = q_tile(m, pfx)(u) # shape = (None, m, mem_dim, 3)
   
    M_q = cbp_M_q(m)([M, u_]) # shape = (None, m, mem_dim, 3) 

    conv_in = M_q 
    for l in range(conf['n_RN_layers']):
        conv = Convolution2D(3, kernel_size=(conf['r_conv_f_v'], conf['mem_dim']), strides=(conf['r_conv_s_v'], conf['r_conv_s_h']), padding='same',
                        name='r_conv%d'%(l), activation='linear', kernel_regularizer=l2(conf['l2reg']))
        conv_out = conv(conv_in) # shape = (None, c, mem_dim, 3)
        conv_out = Activation(conf['cnnact'])(BatchNormalization()(conv_out))
        conv_in = conv_out

    return conv_out

def score_function(pfx):
    def func(x):
        o, u = x[0], x[1]
        A = []
        for i in range(5):
            A.append(x[i+2])
        alpha = K.random_normal_variable(shape=(1,), mean=0, scale=1, name='alpha')
        dot_prod3 = Lambda(name='dot_prod3_'+pfx, function=lambda x: K.sum(multiply([x[0], x[1]]), keepdims=True, axis=1)) 
        
        scores = []
        for i in range(5):
            scores.append(dot_prod3([(alpha*o + (1-alpha)*u), A[i]])) # shape = (None, 1)
        
        z = Activation('softmax')(concatenate(scores)) # shape = (None, 5)
        return z

    return Lambda(func, output_shape=lambda shape:(shape[0],)+(5,))

def answer_selection(M_r, u, A, pfx='anssel'):
    # input shape
    # M_r: (None, c, mem_dim, 3)
    # u: (None, mem_dim)
    # A: [(None, mem_dim), (None, mem_dim), (None, mem_dim), (None, mem_dim), (None, mem_dim)]
    
    c = M_r._keras_shape[1]
    u_ = q_tile(c, pfx)(u) # shape = (None, c, mem_dim, 3)

    dot_prod1 = Lambda(name='dot_prod1_'+pfx, function=lambda x: K.sum(multiply([x[0], x[1]]), axis=2))
    p = dot_prod1([M_r, u_]) # shape = (None, c, 3)
    p = Activation('softmax')(Reshape((c*3,))(p)) # shape = (None, c*3)
    p = RepeatVector(conf['mem_dim'])(p) # shape = (None, mem_dim, c*3)
    

    permute = Lambda(name='permute_'+pfx, function=lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), output_shape=lambda shape:(shape[0],)+(shape[2],)+(shape[1],)+(shape[3],))
    u_ = permute(u_) # shape = (None, mem_dim, c, 3)
    u_ = Reshape((conf['mem_dim'], c*3))(u_) # shape = (None, mem_dim, c*3)
    dot_prod2 = Lambda(name='dot_prod2_'+pfx, function=lambda x: K.sum(multiply([x[0], x[1]]), axis=2))
    o = dot_prod2([p, u_]) # shape = (None, mem_dim)

    z = score_function(pfx)([o, u, A[0], A[1], A[2], A[3], A[4]])

    return z

def build_model():
    # input embedding
    input_nodes, u, E, A = embedding()

    # write networks
    M = write_networks(E) # output shape=(None, m, mem_dim, 3)

    # read networks
    M_r = read_networks(M, u) # output shape=(None, c, mem_dim, 3)

    # answer selection
    output_nodes = answer_selection(M_r, u, A) # output shape=(None, 5)

    #with tf.device("/cpu:0"):
    model = Model(inputs=input_nodes, outputs=output_nodes)

    #model = multi_gpu_model(model, gpus=2)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=conf['lr']))
    return model

def train_and_eval(runid):
    print('Model')
    model = build_model()
    #print(model.summary())
    if conf['pretrained_model'] is not None:
        print ('Using pretrained model %s' % (conf['pretrained_model']))
        with open('cp/weights-'+conf['pretrained_model']+'-bestval.pickle', 'rb') as f:
            w = pickle.load(f)
        model.set_weights(w)
    
    print('Training')
    fit_model(model, weightsf='weights-'+runid+'-bestval.h5')
    #model.save_weights('cp/weights-'+runid+'-final.h5', overwrite=True)
    if conf['pickle_load_weights']:
        with open('cp/weights-'+runid+'-bestval.pickle', 'rb') as f:
            w = pickle.load(f)
        model.set_weights(w)
    else:
        model.load_weights('cp/weights-'+runid+'-bestval.h5')
    
    print('Predict&Eval (best val epoch)')
    return eval(model)

def fit_model(model, **kwargs):
    epochs = conf['epochs']
    callbacks = fit_callbacks(kwargs.pop('weightsf'))
    
    label_tr = np.array([inp_tr['label']]).reshape(-1)
    label_tr = np.eye(5)[label_tr]
    label_val = np.array([inp_val['label']]).reshape(-1)
    label_val = np.eye(5)[label_val]
    
    return model.fit(inp_tr, y=[label_tr], validation_data=[inp_val,
        [label_val]], callbacks = callbacks, epochs=epochs, batch_size=conf['batch_size'])

def fit_callbacks(weightsf):                                  
    return [utils.AnsSelCB(inp_val['q'], inp_val['story'],
        inp_val['a_0'], inp_val['a_1'], inp_val['a_2'], inp_val['a_3'], inp_val['a_4'], inp_val['qid'], inp_val['label'], inp_val, runid),
            ModelCheckpointPickle('cp/'+weightsf, save_best_only=True, monitor='acc', mode='max', save_weights_only = True),
            EarlyStopping(monitor='acc', mode='max', patience=conf['patience'])]

def eval(model):
    for idx, inp in enumerate([inp_val, inp_test]):
        if inp is None:
            res.append(None)
            continue

        label = model.predict(inp, batch_size=conf['batch_size'])

        if idx == 0:
            set_name='val'
            val_acc = utils.eval_QA(label, inp['qid'], inp['label'], set_name, runid)
        else:
            set_name='test'
            test_acc = utils.eval_QA(label, inp['qid'], inp['label'], set_name, runid)

    return (val_acc, test_acc)

if __name__ == "__main__":
    global runid
    params = []
   
    conf, ps, h = config()
    trainf = conf['trainf']
    valf = conf['valf']
    testf = conf['testf']

    if conf['emb'] == 'Glove':
        print('GloVe')
        emb = utils.GloVe(N=conf['emb_dim'])

    print('Dataset')
    load_data(trainf,valf,testf)
    runid = 'RWMN-%x' % (h)
    print('RunID: %s  (%s)' % (runid, ps))
    
    train_and_eval(runid)

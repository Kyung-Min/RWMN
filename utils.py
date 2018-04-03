from __future__ import print_function

from collections import defaultdict
import numpy as np
import json
from operator import itemgetter

from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

import re
from nltk.corpus import stopwords
from collections import namedtuple

import pickle
from keras.callbacks import Callback
import pdb

class Embedder(object):
    """ Generic embedding interface.

    Required: attributes g and N """

    def map_tokens(self, tokens, ndim=2):
        """ for the given list of tokens, return a list of GloVe embeddings,
        or a single plain bag-of-words average embedding if ndim=1.

        Unseen words (that's actually *very* rare) are mapped to 0-vectors. """
        gtokens = [self.g[t] for t in tokens if t in self.g]
        if not gtokens:
            return np.zeros((1, self.N)) if ndim == 2 else np.zeros(self.N)
        gtokens = np.array(gtokens)
        if ndim == 2:
            return gtokens
        else:
            return gtokens.mean(axis=0)

    def map_set(self, ss, ndim=2):
        """ apply map_tokens on a whole set of sentences """
        return [self.map_tokens(s, ndim=ndim) for s in ss]

    def pad_set(self, ss, spad, N=None):
        """ Given a set of sentences transformed to per-word embeddings
        (using glove.map_set()), convert them to a 3D matrix with fixed
        sentence sizes - padded or trimmed to spad embeddings per sentence.

        Output is a tensor of shape (len(ss), spad, N).

        To determine spad, use something like
            np.sort([np.shape(s) for s in s0], axis=0)[-1000]
        so that typically everything fits, but you don't go to absurd lengths
        to accomodate outliers.
        """
        ss2 = []
        if N is None:
            N = self.N
        for s in ss:
            if spad > s.shape[0]:
                if s.ndim == 2:
                    s = np.vstack((s, np.zeros((spad - s.shape[0], N))))
                else:  # pad non-embeddings (e.g. toklabels) too
                    s = np.hstack((s, np.zeros(spad - s.shape[0])))
            elif spad < s.shape[0]:
                s = s[:spad]
            ss2.append(s)
        return np.array(ss2)

class GloVe(Embedder):
    """ A GloVe dictionary and the associated N-dimensional vector space """
    def __init__(self, N=300, glovepath='glove.6B.%dd.txt'):
        """ Load GloVe dictionary from the standard distributed text file.

        Glovepath should contain %d, which is substituted for the embedding
        dimension N. """
        self.N = N
        self.g = dict()
        self.glovepath = glovepath % (N,)

        with open(self.glovepath, 'r') as f:
            for line in f:
                l = line.split()
                word = l[0]
                self.g[word] = np.array(l[1:]).astype(float)


def hash_params(pardict):
    ps = json.dumps(dict([(k, str(v)) for k, v in pardict.items()]), sort_keys=True)
    h = hash(ps)
    return ps, h


class Vocabulary:
    """ word-to-index mapping, token sequence mapping tools and
    embedding matrix construction tools """
    def __init__(self, sentences, count_thres=1):
        """ build a vocabulary from given list of sentences, but including
        only words occuring at least #count_thres times """

        # Counter() is superslow :(
        vocabset = defaultdict(int)
        for s in sentences:
            for t in s:
                vocabset[t] += 1

        vocab = sorted(list(map(itemgetter(0),
                                filter(lambda k: itemgetter(1)(k) >= count_thres,
                                       vocabset.items() ) )))
        self.word_idx = dict((w, i + 2) for i, w in enumerate(vocab))
        self.word_idx['_PAD_'] = 0
        self.word_idx['_OOV_'] = 1
        print('Vocabulary of %d words' % (len(self.word_idx)))

        self.embcache = dict()

    def embmatrix(self, emb):
        """ generate index-based embedding matrix from embedding class emb
        (typically GloVe); pass as weights= argument of Keras' Embedding layer """
        if str(emb) in self.embcache:
            return self.embcache[str(emb)]
        embedding_weights = np.zeros((len(self.word_idx), emb.N))
        for word, index in self.word_idx.items():
            if index == 0:
                embedding_weights[index, :] = np.zeros(emb.N)
            try:
                embedding_weights[index, :] = emb.g[word]
            except KeyError:
                if index == 0:
                    embedding_weights[index, :] = np.zeros(emb.N)
                else:
                    embedding_weights[index, :] = np.random.uniform(-0.25, 0.25, emb.N)  # 0.25 is embedding SD
        self.embcache[str(emb)] = embedding_weights
        return embedding_weights

    def size(self):
        return len(self.word_idx)

def acc(pred, gt_list, qid_list, f=None, runid=None, best=None):
        n_true = 0
        n_false = 0
        scores = []
        qid = -1
        pr_label = np.argmax(pred, axis=1)
        gt_arr = np.array(gt_list)
        ce_losses = []
        m_losses = []
        for i in range(len(gt_list)):
                gt = gt_list[i]
                pr = pred[i][gt]
                ce_loss = np.log2(pr)*-1
                ce_losses.append(ce_loss)
                negl = []
                for j in range(5):
                    if j==gt:
                        continue
                    else:
                        negl.append(pred[i][j])
                neg = np.max(negl)
                m_loss = np.max([0.05+neg-pr, 0])
                m_losses.append(m_loss)
        mean_ce_loss = np.mean(ce_losses)
        mean_m_loss = np.mean(m_losses)
        acc = float(np.sum(pr_label==gt_arr)) / float(gt_arr.shape[-1])
        if best is not None  and runid is not None and acc > best:
            f = open('results/'+runid+'_val.csv', 'wb')
        if f is not None:
                for idx in range(gt_arr.shape[-1]):
                        f.write('%s,%d,%d,' % (qid_list[idx], pr_label[idx], gt_arr[idx]))
                        f.write('%f,%f,%f,%f,%f\n' % (pred[idx][0], pred[idx][1], pred[idx][2],
                                pred[idx][3], pred[idx][4]))
        return acc, mean_ce_loss, mean_m_loss

def eval_QA(pred, qid_list, gt, set_name, runid):
        with open('results/'+runid+'_'+set_name+'.csv', 'wb') as f:
                acc_, ce_loss_, m_loss_ = acc(pred, gt, qid_list, f=f)
                print('Accuracy: %f' %(acc_))
                print('CE Loss: %f'%(ce_loss_))
                print('Margin Loss: %f'%(m_loss_))
        return acc_

"""
Task-specific callbacks for the fit() function.
"""

class AnsSelCB(Callback):
        """ A callback that monitors ACC after each epoch """
        def __init__(self, val_q, val_s, val_a0, val_a1, val_a2, val_a3, val_a4, val_qid, val_pred, inputs, runid):
                self.val_q = val_q
                self.val_s = val_s
                self.val_a0 = val_a0
                self.val_a1 = val_a1
                self.val_a2 = val_a2
                self.val_a3 = val_a3
                self.val_a4 = val_a4
                self.val_qid = val_qid
                self.val_pred = val_pred
                self.val_inputs = inputs
                self.runid = runid
                self.best_acc = -1

        def on_epoch_end(self, epoch, logs={}):
                pred = self.model.predict(self.val_inputs, batch_size=16)
                acc_, ce_loss_, m_loss_ = acc(pred, self.val_pred, self.val_qid, runid=self.runid, best=self.best_acc)
                print('\tval ACC %f\tval CE_LOSS %f\tMargin_LOSS %f' % (acc_, ce_loss_, m_loss_))
                if acc_ > self.best_acc:
                        self.best_acc = acc_
                logs['acc'] = acc_

class ModelCheckpointPickle(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpointPickle, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            filepath = filepath.split('.h5')[0] + '.pickle'
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch + 1, self.monitor, self.best,
                                 current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            #self.model.save_weights(filepath, overwrite=True)
                            cc = self.model.get_weights()
                            with open(filepath, 'wb') as f:
                                pickle.dump(cc, f)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

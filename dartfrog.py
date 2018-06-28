#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import logging
import argparse
from collections import defaultdict, deque, Counter
from nltk import bigrams
import regex
import numpy as np
import pandas as pd
import json
from tqdm import tqdm


argp = argparse.ArgumentParser (description='Word segmentation tool')
argp.add_argument ('dataset', metavar='DATASET', nargs='?', default=sys.stdin,
                   type=argparse.FileType ('r'),
                   help='input dataset (default=stdin)')
argp.add_argument ('output', metavar='OUTPUT', nargs='?', default=sys.stdout,
                   type=argparse.FileType ('w'),
                   help='output file (default=stdout)')
argp.add_argument ('--verbose', action='store_true',
                   help='print detailed debug information')
argp.add_argument ('--vocab', '-b', metavar='VOCABULARY-FILE',
                   nargs='?', default='vocab.csv',
                   help='path to the vocabulary file')
cmd = argp.add_mutually_exclusive_group (required=True)
cmd.add_argument ('--fit', action='store_true',
                  help='train word segmentation model')
cmd.add_argument ('--transform', action='store_true',
                  help='run word segmentation on raw text')
args = argp.parse_args()


class CJKTokenizer:

  """ A CJK character tokenizer that split a CJK string into
      character tokens.
  """

  def __init__ (self):
    self.r_split = regex.compile (ur"""(__[0-9_A-Za-z]+__)
      |(\p{P}|\p{Nd}+|\p{IsHan}|\p{IsHiragana}|\p{IsKatakana}+|\p{IsHangul})
      |\s+
      """, regex.UNICODE|regex.VERBOSE)

  def __call__ (self, doc, encoding='utf-8'):
    if not isinstance (doc, unicode):
      doc = doc.decode (encoding)
    tokens = [t for t in self.r_split.split (doc) if t]
    return tokens


class WordChunker:

  """ A word chunker that performs CJK word segmentation by using
      statistical features learned from the training set.
  """
  r_atomic = regex.compile (ur'^__[0-9_A-Za-z]+__$', regex.UNICODE)

  def feature_subwords (self, vocab, word):
    vocab[word]['term'] += 1
    tokens = self.tokenizer (word)
    #logging.debug (json.dumps (tokens, ensure_ascii=False))
    if len (tokens) == 1:
      return
    for i in range (1, len (tokens)):
      t = u''.join (tokens[:i])
      vocab[t]['seen'] += 1
      vocab[t]['begin'] += 1
    for i in range (len (tokens) - 1, 0, -1):
      t = u''.join (tokens[i:])
      vocab[t]['seen'] += 1
      vocab[t]['end'] += 1
    for i in range (1, len (tokens) - 1):
      vocab[tokens[i]]['seen'] += 1

  def compute_subwords (self):
    eps = np.finfo (np.float64).eps
    odds_left = (self.vocab['begin'].astype (np.float64) + self.vocab['term']) / (eps + self.vocab['seen'] - self.vocab['begin'] - self.vocab['term'])
    self.vocab['k_left'] = np.reciprocal (1 + np.exp (- odds_left))

    odds_right = (self.vocab['end'].astype (np.float64) + self.vocab['term']) / (eps + self.vocab['seen'] - self.vocab['end'] - self.vocab['term'])
    self.vocab['k_right'] = np.reciprocal (1 + np.exp (- odds_right))

    odds_term = self.vocab['term'] / (eps + self.vocab['seen'] - self.vocab['term'])
    self.vocab['k_term'] = np.reciprocal (1 + np.exp (- odds_term))
    self.mean_term_odds = self.vocab['k_term'].mean() - self.vocab['k_term'].std()
    logging.info (u'mean term odds = %.4f', self.mean_term_odds)

  # a regular pattern that matches both Arabic and Chinese numbers
  r_number = regex.compile (ur"""
    (?P<num>(?:(?:[零○一二三四五六七八九十廿卅\p{Nd}]+|[兩两])
     [百千仟萬万億亿兆]*)+)""", regex.UNICODE|regex.VERBOSE)

  r_number_evidence = regex.compile (ur'([\p{N}\p{P}\p{S}])', regex.UNICODE)

  def reducer_number (self, word):
    if word and (self.r_number_evidence.search (word) or any ([len(s) > 1 for s in self.r_number.findall (word)])):
      return self.r_number.sub (u'__numbers__', word)
    return word

  def __init__ (self, max_word_len=None, min_seen=1, tokenizer=CJKTokenizer(),
                extra_features=[], reducers=[]):
    self.max_word_len = max_word_len or 13
    self.min_seen = min_seen
    self.tokenizer = tokenizer
    self.feature_map = {
      'subwords': (self.feature_subwords, self.compute_subwords)
    }
    self.reducer_map = {
      'number': self.reducer_number
    }
    self.extra_features = extra_features
    self.reducers = reducers

  def fit (self, doc_iterable, encoding='utf-8',
           ws_pattern=ur'\s+'):
    """ Train on whitespace-separated texts.
    """
    r_word_separator = regex.compile (ws_pattern, regex.UNICODE)
    vocab = defaultdict (lambda: defaultdict (int))
    for doc in tqdm (doc_iterable):
      if not isinstance (doc, unicode):
        doc = doc.decode (encoding)
      for word in r_word_separator.split (doc):
        if word:
          #word = unicode.lower (word)
          for reducer in self.reducers:
            word = self.reducer_map[reducer] (word)
          if word:
            vocab[word]['seen'] += 1
            for feature in self.extra_features:
              self.feature_map[feature][0] (vocab, word)
    self.vocab = pd.DataFrame.from_dict (vocab, orient='index').fillna (0).astype (int)
    # compute extra features
    for feature in self.extra_features:
      self.feature_map[feature][1] ()

  def save (self, to_file):
    self.vocab.sort_values (by='seen', ascending=False).to_csv (
         to_file, encoding='utf-8', index_label='word',
         columns=['seen', 'begin', 'end', 'term'])

  def load (self, from_file):
    self.vocab = pd.read_csv (from_file, encoding='utf-8', index_col='word')
    logging.info (u'vocabularies loaded: %s', self.vocab.shape)
    for feature in self.extra_features:
      self.feature_map[feature][1] ()

  # forward maximum match cutter
  def cut_fmm (self, text):
    tokens = deque (u''.join (self.tokenizer (text))) # use char-by-char
    words = deque()
    while tokens:
      w = deque ()
      while tokens and len (u''.join (w)) < self.max_word_len:
        w.append (tokens.popleft())
      while len (w) > 1 and not self.vocab_recall (u''.join (w), 'seen', no_reduce=True):
        tokens.appendleft (w.pop())
      words.append (u''.join (w))
    return words

  # backward maximum match cutter
  def cut_bmm (self, text):
    tokens = deque (u''.join (self.tokenizer (text))) # use char-by-char
    words = deque()
    while tokens:
      w = deque ()
      while tokens and len (u''.join (w)) < self.max_word_len:
        w.appendleft (tokens.pop())
      while len (w) > 1 and not self.vocab_recall (u''.join (w), 'seen', no_reduce=True):
        tokens.append (w.popleft())
      words.appendleft (u''.join (w))
    return words

  def vocab_recall (self, word, attr, no_reduce=False):
    if self.vocab.index.contains (word) and self.vocab.at[word, 'seen'] >= self.min_seen:
      #rv = (self.vocab.at[word, attr] + self.vocab.at[word, 'k_term']) - 1 #/ 2
      rv = self.vocab.at[word, attr] # * 2 - 1
      logging.debug (u'recalled [%s].%s = %.4f', word, attr, rv)
      return rv
    elif self.r_atomic.match (word):
      return 1.0
    elif no_reduce:
      return 0.0

    for reducer in self.reducers:
      word = self.reducer_map[reducer] (word)
    return self.vocab_recall (word, attr, no_reduce=True)

  def sent_recall (self, tokens):
    # compute matrix of word boundary odds
    k = np.zeros ((len (tokens), len (tokens)))
    for i in range (len (tokens)):
      for j in range (len (tokens)):
        if abs (i - j) > self.max_word_len:
          continue
        if i == j:
          k[i,j] = max (self.vocab_recall (tokens[i], 'k_term'), np.finfo (np.float64).eps)
        elif i < j:
          k[i,j] = self.vocab_recall (u''.join (tokens[i:j+1]), 'k_term')
          #k[i,j] = self.vocab_recall (u''.join (tokens[i:j+1]), 'k_left')
        else:
          k[i,j] = self.vocab_recall (u''.join (tokens[j:i+1]), 'k_term')
          #k[i,j] = self.vocab_recall (u''.join (tokens[j:i+1]), 'k_right')
    return k

  # backward maximum odds cutter
  def cut_bmo (self, text):
    tokens = self.tokenizer (text)
    logging.debug (json.dumps (tokens, ensure_ascii=False))

    edges = [0]
    tokens.append (u'')	# force final review at end of text
    k = self.sent_recall (tokens)

    # emit edges & resolve ambiguities
    for n in range (1, len (tokens)):
      fmax = n + k[n,n:].argmax() + 1
      bmax = k[n-1,:n].argmax()
      if tokens[n] and max (k[n,n:].max(), k[n-1,:n].max()) < self.mean_term_odds:
        continue
      #logging.debug (u'cut %s|%s, k[n-1,:n].sum()=%.4f)',
      #               u''.join (tokens[bmax:n]), u''.join (tokens[n:fmax]), k[n-1,:n].sum())

      # 1. remove backward sub-words that has no impact to backward odds sum
      while (len (edges) > 1) and (bmax <= edges[-2]) and (k[n-1,:n].sum() >= k[n-1,edges[-1]:n].sum()):
        #logging.debug (u'un-cut [%s]', u''.join (tokens[edges[-2]:n]))
        edges.pop()

      # 2. resolve possible ambiguities: move / merge
      def score_cut (c):
        return k[c-1,:c].sum() + k[c,c:].sum()
      while (len (edges) > 1) and (bmax >= edges[-2]) and (bmax < edges[-1]):
        #logging.debug (u'review-cut [%s|%s]', u''.join (tokens[edges[-2]:edges[-1]]), u''.join (tokens[edges[-1]:n]))
        old_cut = score_cut (edges[-1])
        move_cut = score_cut (bmax)
        merge_cut = k[n-1,edges[-2]:n].sum() + (1 - k[n-1,edges[-1]:n].max())
        if move_cut >= max (old_cut, merge_cut):
          edges[-1] = bmax
          #logging.debug (u'modified-cut [%s|%s] old=%.4f, move=%.4f, merge=%.4f',
          #               u''.join (tokens[edges[-2]:edges[-1]]), u''.join (tokens[edges[-1]:n]), old_cut, move_cut, merge_cut)
        elif merge_cut >= max (old_cut, move_cut):
          edges.pop()
          #logging.debug (u'merged [%s]', u''.join (tokens[edges[-1]:n]))
        else:
          break
      edges.append (n)

    # generate word list from edges
    words = [u''.join (tokens[a:b]) for a, b in bigrams (edges)]
    return words

  def transform (self, text_iterable, to_file, encoding='utf-8'):
    for text in tqdm (text_iterable):
      words = self.cut_bmo (text)
      print (u' '.join (words).encode (encoding), file=to_file)


def main (_):
  if args.verbose:
    logging.basicConfig (level=logging.DEBUG)
  else:
    logging.basicConfig (level=logging.INFO)

  chunker = WordChunker (extra_features=['subwords'], reducers=['number'], min_seen=1)

  if args.fit:
    # train the model
    chunker.fit (args.dataset)
    chunker.save (args.vocab)
  elif args.transform:
    # transform input text into segmented words
    chunker.load (args.vocab)
    chunker.transform (args.dataset, args.output)


if __name__ == '__main__':
  main (None)


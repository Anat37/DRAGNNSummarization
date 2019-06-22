import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.cuda import FloatTensor, LongTensor
import random

import struct
import os
import json
from tensorflow.core.example import example_pb2

VOCAB_SIZE = 50000
ADDITIONAL_WORDS = 400

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

PAD_TOKEN_ID = 1
UNKNOWN_TOKEN_ID = 0
START_DECODING_ID = 2
STOP_DECODING_ID = 3

class Vocab(object):

  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    # Read the vocab file and add words up to max_size
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
          continue
        w = pieces[0]
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
          break

    print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

  def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    return self._count

  def write_metadata(self, fpath):
    print("Writing word embedding metadata file to %s..." % (fpath))
    with open(fpath, "w") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in xrange(self.size()):
        writer.writerow({"word": self._id_to_word[i]})
        


def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words.split():
        w = str(w)
        i = vocab.word2id(w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            if oov_num < ADDITIONAL_WORDS:
                ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
            else:
                ids.append(i)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words.split():
        if w == SENTENCE_START or w == SENTENCE_END:
            continue
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
                if vocab_idx < VOCAB_SIZE + ADDITIONAL_WORDS:
                    ids.append(vocab_idx)
                else:
                    ids.append(i)
            else:
                ids.append(i)
        else:
            ids.append(i)
    return ids

def outputids2words(id_list, vocab, article_oovs):
    words = []
    oov_len = len(article_oovs)
    for i in id_list:
        if i < VOCAB_SIZE:
            w = vocab.id2word(i)
        else:
            article_oov_idx = i - vocab.size()
            #print('unknown generated')
            if article_oov_idx < oov_len:
                w = article_oovs[article_oov_idx]
        words.append(w)
    return ' '.join(words)


def example_gen(filename):
    reader = open(filename, 'rb')
    examples = []
    while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        e = example_pb2.Example.FromString(example_str)
        examples.append(e)
        
    for e in examples:  
        article_text = e.features.feature['article'].bytes_list.value[0]
        abstract_text = e.features.feature['abstract'].bytes_list.value[0]
        yield (article_text.decode('utf-8'), abstract_text.decode('utf-8'))
        
        
def add_padding(articles):
    lens = [len(article) for article in articles]
    max_len = max(lens)
    
    for i in range(len(articles)):
        articles[i].extend([PAD_TOKEN_ID]*(max_len - len(articles[i])))
    return np.array(articles).T

def add_padding_for_tagging(articles, targets):
    lens = [len(article) for article in articles]
    max_len = max(lens)
    
    for i in range(len(articles)):
        targets[i].extend([0]*(max_len - len(articles[i])))
        articles[i].extend([PAD_TOKEN_ID]*(max_len - len(articles[i])))
    return np.array(articles).T, np.array(targets)

def calculate_mask(articles):
    mask = (articles == PAD_TOKEN_ID)
    mask = np.logical_xor(mask, np.ones(articles.shape))
    return np.array(mask, dtype=np.int32)

def get_target(self, article, abstract):
    return [ int(i in abstract and i != UNKNOWN_TOKEN_ID) for i in article]

class Batcher():
    
    def __init__(self, vocab, filename, batch_size, max_article_len, max_target_len):
        self.batch_size = batch_size
        generator = example_gen(filename)

        self.batches = []
        unknown_words_cnt = 0
        self.articles = []
        self.targets = []
        self.unknown_words = []
        self.decoder_inputs = []
        while True:
            articles = []
            targets = []
            unknown_words = []
            decoder_inputs = []
            for i in range(batch_size):
                try:
                    article_text, abstract_text = next(generator)
                    #print(article_text)
                    article_ids, unknown_words_list = article2ids(article_text, vocab)
                    art_len = min(max_article_len, len(article_ids))
                    article_ids = article_ids[:art_len]
                    
                    target = abstract2ids(abstract_text, vocab, unknown_words_list)
                    tar_len = min(max_target_len, len(target))
                    target.append(STOP_DECODING_ID)
                    target = target[:tar_len]
                    decoder_input = [START_DECODING_ID]
                    decoder_input.extend(abstract2ids(abstract_text, vocab, unknown_words_list))
                    decoder_input = decoder_input[:tar_len]
                    articles.append(article_ids)
                    targets.append(target)
                    decoder_inputs.append(decoder_input)
                    unknown_words.append(unknown_words_list)
                    unknown_words_cnt = max(len(unknown_words_list), unknown_words_cnt)
                    
                except:
                    break
            if len(articles) == 0:
                break
            self.articles.extend(articles)
            self.targets.extend(targets)
            self.unknown_words.extend(unknown_words)
            self.decoder_inputs.extend(decoder_inputs)
            articles = add_padding(articles)
            targets = add_padding(targets)
            decoder_inputs = add_padding(decoder_inputs)
            target_mask = calculate_mask(targets)
            encoder_mask = calculate_mask(articles)
            self.batches.append( (articles, targets, encoder_mask, target_mask, decoder_inputs) )
        print(len(self.batches))
        print(unknown_words_cnt)
    
    def generator(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
            
    def get_random_sample(self):
        i = random.randint(0, len(self.articles) - 1)
        return np.array([self.articles[i]]).T, np.array(self.targets[i]), np.array([self.decoder_inputs[i]]).T, self.unknown_words[i]
    

class CachedBatcher():
    
    def __init__(self, vocab, filename, batch_size, max_article_len, max_target_len):
        self.cache_filename = 'cached' + str(max_article_len) + '_' + str(max_target_len) + '/' + filename
        self.vocab = vocab
        self.batch_size = batch_size
        
        
        if not os.path.isfile(self.cache_filename):
            self.create_cache(filename, max_article_len, max_target_len)
        
    def create_cache(self, filename, max_article_len, max_target_len):
        
        generator = example_gen(filename)
        
        if not os.path.exists(os.path.dirname(self.cache_filename)):
            try:
                os.makedirs(os.path.dirname(self.cache_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
        with open(self.cache_filename, 'w+') as f:
            while True:            
                try:
                    article_text, abstract_text = next(generator)
                    #print(article_text)
                    article_ids, unknown_words_list = article2ids(article_text, self.vocab)
                    art_len = min(max_article_len, len(article_ids))
                    article_ids = article_ids[:art_len]
                    article_ids_unk = [i if i < VOCAB_SIZE else UNKNOWN_TOKEN_ID for i in article_ids]
                    target = abstract2ids(abstract_text, self.vocab, unknown_words_list)
                    tar_len = min(max_target_len, len(target))
                    target.append(STOP_DECODING_ID)
                    target = target[:tar_len]
                
                    decoder_input = [START_DECODING_ID]
                    decoder_input.extend(abstract2ids(abstract_text, self.vocab, unknown_words_list))
                    decoder_input = decoder_input[:tar_len]
                    decoder_input = [i if i < VOCAB_SIZE else UNKNOWN_TOKEN_ID for i in decoder_input]

                    sample = (article_ids, article_ids_unk, target, decoder_input, unknown_words_list)
                    f.write(json.dumps(sample) + '\n')
                except:
                    break

                
            
    def load_cache(self):
        samples = []
        with open(self.cache_filename) as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)
        
        random.shuffle(samples)
        ind = 0
        self.batches = []
        unknown_words_cnt = 0
        self.articles = []
        self.targets = []
        self.unknown_words = []
        self.decoder_inputs = []
        self.articles_ids_unk = []
        while True:
            articles = []
            articles_ids_unk = []
            targets = []
            unknown_words = []
            decoder_inputs = []
            for i in range(self.batch_size):
                if ind >= len(samples):
                    break
                article_ids, article_ids_unk, target, decoder_input, unknown_words_list = samples[ind]
                ind += 1
               
                articles.append(article_ids)
                articles_ids_unk.append(article_ids_unk)
                targets.append(target)
                decoder_inputs.append(decoder_input)
                unknown_words.append(unknown_words_list)
                unknown_words_cnt = max(len(unknown_words_list), unknown_words_cnt)
                
               
            if len(articles) == 0:
                break
            self.articles.extend(articles)
            self.articles_ids_unk.extend(articles_ids_unk)
            self.targets.extend(targets)
            self.unknown_words.extend(unknown_words)
            self.decoder_inputs.extend(decoder_inputs)
            articles = add_padding(articles)
            articles_ids_unk = add_padding(articles_ids_unk)
            targets = add_padding(targets)
            decoder_inputs = add_padding(decoder_inputs)
            target_mask = calculate_mask(targets)
            encoder_mask = calculate_mask(articles)
            self.batches.append( (articles, articles_ids_unk, targets, encoder_mask, target_mask, decoder_inputs) )
        print(len(self.batches))
        print(unknown_words_cnt)        
        
    
    def generator(self):
        self.load_cache()
        for batch in self.batches:
            yield batch
            
    def get_random_sample(self):
        i = random.randint(0, len(self.articles) - 1)
        return np.array([self.articles[i]]).T, np.array([self.articles_ids_unk[i]]).T, np.array(self.targets[i]), np.array([self.decoder_inputs[i]]).T, self.unknown_words[i]
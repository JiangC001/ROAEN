'''
Description:  
Author: J chen
'''
import os
import sys
import re
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset

def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        # print(data)
        # print('*'*100)
        a=0
        for d in data:

            #  aspects [{'term': ['Boot', 'time'], 'from': 0, 'to': 2, 'polarity': 'positive'}]
            for aspect in d['aspects']:
                a +=1
                text_list = list(d['token'])  #  token ['Boot', 'time', 'is', 'super', 'fast', ',', 'around', 'anywhere', 'from', '35', 'seconds', 'to', '1', 'minute', '.']
                tok = list(d['token'])
                length = len(tok)
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(
                    tok)  #  boot time is super fast , around anywhere from 35 seconds to 1 minute .
                asp = list(aspect['term'])
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']
                pos = list(d['pos'])
                head = list(d['head'])
                deprel = list(d['deprel'])  # deprel
                # short = list(d['short'])

                aspect_post = [aspect['from'], aspect['to']]
                post = [i - aspect['from'] for i in range(aspect['from'])] \
                       + [0 for _ in range(aspect['from'], aspect['to'])] \
                       + [i - aspect['to'] + 1 for i in range(aspect['to'], length)]
                post_1 = [1 for i in range(len(head))]

                #Reserved tree
                adj_f = np.zeros((len(post_1),len(post_1)))
                adj_b = np.ones((len(post_1),len(post_1)))
                adj_f_aspect = np.zeros((len(post_1),len(post_1)))
                adj_b_aspect = np.zeros((len(post_1),len(post_1)))

                for i,j in enumerate(head):
                    adj_f[i,i], adj_b[i,i] = 1, 0
                    adj_f[i,int(j)-1], adj_b[i,int(j)-1] = 1, 0
                    adj_f[int(j)-1,i], adj_b[int(j)-1,i] = 1, 0
                    if i in list(range(aspect_post[0],aspect_post[1])):
                        adj_f_aspect[i,i] = 1
                        adj_f_aspect[i,int(j)-1] = 1
                        adj_f_aspect[int(j)-1,i] = 1

                for i,j in enumerate(head):
                    for k in range(len(head)):
                        if i in list(range(aspect_post[0],aspect_post[1])):
                            adj_b_aspect[i,k] = 1
                            adj_b_aspect[k,i] = 1
                            adj_b_aspect[i,int(j)-1] = 0
                            adj_b_aspect[int(j)-1,i] = 0
                            adj_b_aspect[i,i] = 0

                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]
                else:
                    # [0,0,0,0,1,1,1,0,0,...,0]
                    mask = [0 for _ in range(aspect['from'])] \
                           + [1 for _ in range(aspect['from'], aspect['to'])] \
                           + [0 for _ in range(aspect['to'], length)]

                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'post_1':post_1,'head': head, \
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list,'adj_f':adj_f,'adj_b':adj_b,\
                          'adj_f_aspect':adj_f_aspect,'adj_b_aspect':adj_b_aspect}

                all_data.append(sample)
        # print('a:',a)
    return all_data

def build_tokenizer(fnames, max_length, data_file):

    parse = ParseData

    #pickle.load(file)
    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length, parse=parse)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer

class Vocab(object):
    ''' vocabulary of dataset '''
    def __init__(self, vocab_list, add_pad, add_unk):

        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0

        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = self._length #0
            self._length += 1 #length+1=1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1

        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, id_):   
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return self._length

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

class Tokenizer(object):
    ''' transform text to indices '''
    def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

        self.pos_char_to_int = pos_char_to_int
        self.pos_int_to_char = pos_int_to_char

    @classmethod
    def from_files(cls, fnames, max_length, parse, lower=True):
        corpus = set()
        pos_char_to_int, pos_int_to_char = {}, {}
        for fname in fnames:
            for obj in parse(fname):
                text_raw = obj['text']
                if lower:
                    text_raw = text_raw.lower()
                corpus.update(Tokenizer.split_text(text_raw))

        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower, pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)

    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        # [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
        x = (np.zeros(maxlen) + pad_id).astype(dtype)

        #[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        else:
            trunc = sequence[:maxlen]
        #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]→[0 1 2 3 4 5 6 7 8 9]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc

        return x
    
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = Tokenizer.split_text(text)
        sequence = [self.vocab.word_to_id(w) for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()

        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length, 
                                      padding=padding, truncating=truncating)

    @staticmethod
    def split_text(text):

        return text.strip().split()

class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, tokenizer, opt, vocab_help):

        parse = ParseData
        post_vocab, pos_vocab, dep_vocab, pol_vocab = vocab_help
        data = list()
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname),
                        total=len(parse(fname)),
                        desc="Training examples"):
            text = tokenizer.text_to_sequence(obj['text'])
            aspect = tokenizer.text_to_sequence(obj['aspect'])
            post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]
            post = tokenizer.pad_sequence(post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]
            pos = tokenizer.pad_sequence(pos, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]
            deprel = tokenizer.pad_sequence(deprel, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')

            mask = tokenizer.pad_sequence(obj['mask'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')

            adj = np.ones(opt.max_length) * opt.pad_id

            if opt.parseadj:

                from DualGCN_ABSA_main.LAL_Parser.src_joint.absa_parser import headparser

                # * adj
                headp, syntree = headparser.parse_heads(obj['text'])

                adj = softmax(headp[0])
                adj = np.delete(adj, 0, axis=0)
                adj = np.delete(adj, 0, axis=1)
                adj -= np.diag(np.diag(adj))
                if not opt.direct:
                    adj = adj + adj.T
                adj = adj + np.eye(adj.shape[0])
                adj = np.pad(adj, (0, opt.max_length - adj.shape[0]), 'constant')

            if opt.parsehead:
                from DualGCN_ABSA_main.LAL_Parser.src_joint.absa_parser import headparser

                headp, syntree = headparser.parse_heads(obj['text'])

                syntree2head = [[leaf.father for leaf in tree.leaves()] for tree in syntree]
                head = tokenizer.pad_sequence(syntree2head[0], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            else:
                head = tokenizer.pad_sequence(obj['head'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            length = obj['length']
            polarity = polarity_dict[obj['label']]

            data.append({
                'text': text, 
                'aspect': aspect, 
                'post': post,
                'pos': pos,
                'deprel': deprel,
                'head': head,
                'adj': adj,
                'mask': mask,
                'length': length,
                'polarity': polarity,
            })

        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

def _load_wordvec(data_path, embed_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        # 200维Glove向量
        if embed_dim == 200:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>' or tokens[0] == '<unk>':  # avoid them
                    continue
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        elif embed_dim == 300:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>':
                    continue
                elif tokens[0] == '<unk>':
                    word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
                word = ''.join((tokens[:-300]))
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[word] = np.asarray(tokens[-300:], dtype='float32')
        else:
            print("embed_dim error!!!")
            exit()
            
        return word_vec #{'word':vector}

def build_embedding_matrix(vocab, embed_dim, data_file):

    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))

    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = _load_wordvec(fname, embed_dim, vocab)

        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix

def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id #cls
        self.sep_token_id = self.tokenizer.sep_token_id #sep
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}

        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):

            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]

            text_list = obj['text_list']
            text_len = len(text_list)

            #把adj拿出来
            adj_f = np.pad(obj['adj_f'],(0,opt.max_length-len(text_list)),'constant')
            adj_f_aspect = np.pad(obj['adj_f_aspect'],(0,opt.max_length-len(text_list)),'constant')
            adj_b = np.pad(obj['adj_b'],(0,opt.max_length-len(text_list)),'constant')
            adj_b_aspect = np.pad(obj['adj_b_aspect'],(0,opt.max_length-len(text_list)),'constant')

            post_1 = obj['post_1']+[0]*(opt.max_length-len(text_list))

            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end: ]

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left): #['i','am','good']
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)
                    left_tok2ori_map.append(ori_i)
            asp_start = len(left_tokens)
            offset = len(left)

            for ori_i, w in enumerate(term):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term)

            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i+offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len-2*len(term_tokens) - 3:#为什么要*2呢
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()


            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens)
            tok_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')

            context_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(
                bert_tokens)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(term_tokens)+[tokenizer.sep_token_id]

            context_asp_len = len(context_asp_ids)
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)
            context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            #[0,1,1,1,...,1,0,0,0,...,0]
            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)

            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            #[0,0,0,...,0,1,1,...,1,0,...,0]
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]

            while len(aspect_mask) > 85:
                aspect_mask.pop(0)

            context_asp_attention_mask = [1] * context_asp_len + paddings
            context_asp_ids += paddings

            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            post_1 = np.asarray(post_1, dtype='int64')
            adj_f = np.asarray(adj_f,dtype='float32')
            adj_b = np.asarray(adj_b,dtype='float32')
            adj_f_aspect = np.asarray(adj_f_aspect,dtype='float32')
            adj_b_aspect = np.asarray(adj_b_aspect,dtype='float32')


            data = {
                'adj_f':adj_f, #adj
                'adj_b':adj_b, #r_adj
                'adj_f_aspect':adj_f_aspect, #aspect adj
                'adj_b_aspect':adj_b_aspect, #r_aspect adj
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'text_len':text_len,
                'post_1':post_1,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,

            }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

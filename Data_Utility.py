from torch.utils.data import Dataset
from torch import tensor
import torch
import numpy as np
import pickle
import os

def get_vocabulary(fnames,dat_fname,max_seq_len):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        vocabulary = pickle.load(open(dat_fname, 'rb'))
    else:
        vocabulary = Vocabulary(max_seq_len=max_seq_len)
        for fname in fnames:
            vocabulary.take_into_vocabulary_form_text(text_fileName=fname)
        pickle.dump(vocabulary, open(dat_fname, 'wb'))
    return vocabulary

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class Vocabulary(object):
    """
    This class is mainly used to build vocabulary so that the token in the data set has a corresponding index
    """
    def __init__(self,max_seq_len,lower=True):
        self.max_seqLen = max_seq_len
        self.lower = lower
        self._token2idx = {}
        self._idx2token = {}

    def take_into_vocabulary_form_text(self,text_fileName):
        """
        Separate all tokens from text and assign index to them
         :param text: The text information to be processed is a string type, and the format is specified as all tokens are separated by ''
        """
        text = ""
        file = open(text_fileName,'r',newline='\n',errors='ignore')
        lines = file.readlines()
        file.close()
        for i in range(0,len(lines),3):
            sen_left,_,sen_right = (s.lower().strip() for s in lines[i].partition("$T$"))
            aspect = lines[i+1].lower().strip()
            text_raw = sen_left + " " + aspect + " " + sen_right
            text += text_raw + " "
        if self.lower:
            text = text.lower()
        tokens = text.split()
        for token in tokens:
            self.add_token(token)

    def add_token(self,token):
        """
        Add a token to vocabulary
         :param token: token to be added
         :return: return the index corresponding to this token
        """
        exist_flag,index = self.lookup_index(token)
        if not exist_flag:
            self._token2idx[token] = index
            self._idx2token[index] = token
        return index

    def lookup_index(self,token):
        if self.lower:
            token = token.lower()
        if token in self._token2idx.keys():
            return True,self._token2idx[token]
        else:
            return False,len(self._token2idx)+1

    def row_to_indexs(self,text_row,reverse = False,padding='post', truncating='post'):
        """
        Convert a sentence into an index vector
         :param text_row: a sentence (or document in NLP)
         :param vocabulary: vocabulary list (in order to find the index corresponding to each word)
         :return: index vector
        """
        if self.lower:
            text_row = text_row.lower()
        words = text_row.strip().split()
        indexs = [self.lookup_index(word)[1] for word in words]
        if len(indexs) == 0:
            indexs = [0]
        if reverse:
            indexs = indexs[::-1]
        indexs = pad_and_truncate(sequence=indexs,maxlen=self.max_seqLen,padding=padding, truncating=truncating)
        return indexs

    def get_size(self):
        return len(self._token2idx)

    def get_w2i_i2w(self):
        return self._token2idx,self._idx2token

class Vectorizer(object):
    """
    Give the word embed in Vocabulary a vector matrix
    """
    def __init__(self,vocabulary,embedding_dim):
        super(Vectorizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._vocabulary = vocabulary
        self._word2idx = vocabulary.get_w2i_i2w()[0]

    def get_embeddingMatrix(self):
        return self._embedding_matrix

    def generate_embeddingMatrix(self,embedding_fileName,dat_fname):
        """
        Generate an embedding matrix -> each row vector represents the embedding of a word, and the subscript of the row vector represents the word (vocabulary
         Index in word2idx); if there is no corresponding
         :param embedding_fileName:file name -> The storage format in this file is: {word:vecter}
         :return: embedding matrix
        """
        if os.path.exists(dat_fname):
            print('loading embedding_matrix:', dat_fname)
            self._embedding_matrix = pickle.load(open(dat_fname, 'rb'))
        else:
            print('loading word vectors...')
            self._embedding_matrix = np.zeros((self._vocabulary.get_size() + 2, self._embedding_dim))
            word_vec = self._load_word_vec(embedding_fileName)
            print('building embedding_matrix:', dat_fname)
            for word, i in self._word2idx.items():
                vec = word_vec.get(word)
                if vec is not None:
                    # words not found in embedding index will be all-zeros.
                    self._embedding_matrix[i] = vec
            pickle.dump(self._embedding_matrix, open(dat_fname, 'wb'))
        return self._embedding_matrix

    def _load_word_vec(self,fname):
        fin = open(fname,'r',encoding='utf-8',newline='\n',errors='ignore')
        word_vec = {}
        for line in fin:
            tokens = line.rstrip().split()
            word, vec = ''.join(tokens[:-self._embedding_dim]), tokens[-self._embedding_dim:]
            if word in self._word2idx.keys():
                word_vec[word] = np.asarray(vec, dtype='float32')
        return word_vec

class My_DataSet(Dataset):
    def __init__(self,data_fileName,vocabulary,mode):
        """
        Construct dataset -> a sentence is represented by the subscript of each word in embedding_matrix
         For example: "I love you", the corresponding subscripts of i, love, and you in embedding_matrix are 1, 2, 3.
             Then this sentence can be expressed as [1,2,3]
         :param data_fileName: The name of the file (path) where the text data to be predicted is stored
         :param vocabulary: Vocabulary class object constructed with the word used this time
         :param mode: select the type, you can choose LSTM, TD-LSTM and TC-LSTM, each has a different processing method
        """
        self.dataset = []
        file = open(data_fileName,'r',newline='\n',errors='ignore')
        lines = file.readlines()
        file.close()
        if mode == "IAN":
            for i in range(0,len(lines),3):
                sen_left,_,sen_right = (s.lower().strip() for s in lines[i].partition("$T$"))
                aspect = lines[i+1].lower().strip()
                polarity = lines[i+2].strip()
                if polarity == 'positive':
                    polarity = 1
                elif polarity == 'negative' :
                    polarity = -1
                else :
                    polarity = 0

                text_row = sen_left + " " + sen_right
                sample = {'context':vocabulary.row_to_indexs(text_row),
                          'aspect':vocabulary.row_to_indexs(aspect),
                          'y':tensor(int(polarity)+1,dtype=torch.long)}   # +1 is because of the targets parameter of CrossEntropyLoss,

                self.dataset.append(sample)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
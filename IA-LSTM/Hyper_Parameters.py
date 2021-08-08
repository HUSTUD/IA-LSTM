import torch
import logging
import sys

#Use the class as a namespace
class opt:
    #file_args:
    #dataset = 'restaurant'
    dataset = 'AR_HOTEL'
    data = {

            'HAAD':
                {'train_path': "./Dataset/semeval14/HAAD_Ttain.txt",
                 'test_path': "./Dataset/semeval14/HAAD_Test_G.txt"},
            'AR_HOTEL':
                {'train_path': "./Dataset/SemEval2016_arabic/HOTEL_Ttain.txt",
                 'test_path': "./Dataset/SemEval2016_arabic/HOTEL_Test.B.txt"}
            }
    train_path,test_path = data[dataset].values() # the path of traning data & the path of test data
    embedding_name = '300_84'
    embedding_set = {
            '100_27':"./Data/glove.twitter.27B.100d.txt",
            #'300_84':"./Word_Embedding/Glove/glove.840B.300d.txt",
            #'300_84': "./Data/glove.840B.300d.txt",
            '300_84': "./Data/tweets_cbow_300.txt",
            '300_42':""}
    embedding_file_path = embedding_set[embedding_name]

    #preTrain_args:                #Main hyperparameters
    train_ratio = 1            #the size ratio of train
    validation_ratio = 0       #the size ratio of validation
    embedding_dim = 300        #dimension of word embedding
    max_seq_len = 85
    paded_mark = 0
    seed = int(1234)
    uniform_range = 0.1
    l2reg = 1e-5
    dropout = 0.5

    #model_args:
    model_name = 'IAN'
    input_dim = embedding_dim
    hidden_dim = 300
    num_layer = 1
    bias = True
    batch_first = True
    num_class = 3
    hops = 3

    #train_args:
    epoch = 30
    learn_rate = 1e-4       #learning rate
    batch_size = 16         #the mini-batch size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    patience = 5

#Preprocessing
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
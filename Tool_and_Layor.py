from torch import nn
import torch
import torch.nn.functional as F
import math

class Dynamic_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.5,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).
        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(Dynamic_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        #self.rnn_type = rnn_type

        #if self.rnn_type == 'LSTM':
        self.RNN = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        # elif self.rnn_type == 'GRU':
        #     self.RNN = nn.GRU(
        #         input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        #         bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        # elif self.rnn_type == 'RNN':
        #     self.RNN = nn.RNN(
        #         input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        #         bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self,x,x_len):
        """
        Use lstm to process variable length sequences
         :param x: input after padding
         :param x_len: Record the length of each seqence in x (that is, the number of words in each line after removing the padding mark)
         :return: output of lstm layor (including output, ht, ct)
        """
        # 1.1_pack paded sequences：
        # sort: Because pytorch requires long sentences to be on top of paded sequences
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        # Reason for setting: Because x will change position according to x_sort_idx (the longest is first), after the change, the subscript of each vector in x will also be rearranged, then we
        # Need to know what the original x was like. Of course, this is also due to the problem that python assignment is also a reference, otherwise a new variable will be used to store the original letter.
        # Interest is fine.
        # Sentence explanation: If x: 3 1 2 4 -> 1 2 3 4
        # indices：0 1 2 3 -> x_sort_id: 1 2 0 3
        # (Because the subscript has also changed after the arrangement) indices: 0 1 2 3
        # ------------------------------------------------- ---------------------------------
        # We know that if the original x is labeled according to the indices of the new x, it should be like this:
        # new_x: 3 1 2 4 -> From this we can find -> After sorting x_sort_id again, its subscript is 2 0 1 3
        # indices: 2 0 1 3
        # ------------------------------------------------- ---------------------------------
        # !!! intuitive explain:
        # Because x_sort_idx uses old_indices to record the sorted order of old_x, and it is arranged in order before x changes (0, 1...) (nonsense)
        # So, look at x_sort_idx and new_x together, and indices can be used as the whole indices. Intuitively we can easily see that
        # x_sort_idx After sorting, it corresponds to the arrangement order of old_x, so we sort it, and the indices obtained at this time are naturally new_x
        # Indices when arranged in the order of old_x
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        # 1.2_pack
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        # 1.3 process using LSTM
        # if self.rnn_type == 'LSTM':
        out_pack, (ht, ct) = self.RNN(x_emb_p, None)    #ht.shape = ct.shape = [num_layor,batch_size,hidden_num]
        # else:
        #     out_pack, ht = self.RNN(x_emb_p, None)
        #     ct = None
        #1.4_unsort
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)       #Because we mainly use ht batch_size, hidden_num part, so change num_layor to
                                              #The first dimension is easy to choose
        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            #if self.rnn_type == 'LSTM':
            ct = torch.transpose(ct, 0, 1)[
                x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            ct = torch.transpose(ct, 0, 1)
            return out, (ht, ct)

def squeeze_sequence(x, x_len,batch_first = True):
    """
    Because the length of the original padding sequence group is artificially set, it may appear that the length of these seqs is less than the preset value
     -> The length of the compressed sequence is the length of the longest seq in the sequence group.
     :param x: sequence group
     :param x_len: the number of each sequence
     :return: new x
    """
    """sort"""
    x_sort_idx = torch.sort(-x_len)[1]
    x_unsort_idx = torch.sort(x_sort_idx)[1]
    x_len = x_len[x_sort_idx]
    x = x[x_sort_idx]
    """pack"""
    x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=batch_first)
    """unpack: out"""
    out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p,
                                                 batch_first=batch_first)  # (sequence, lengths)
    out = out[0]  # (sequence, lengths)
    """unsort"""
    out = out[x_unsort_idx]
    return out

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
             hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.proj = nn.Linear(embed_dim,out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        if self.n_head != q_len:
            raise RuntimeError('n_head != query num')
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        k = k.float()
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)
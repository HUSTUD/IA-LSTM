from Tool_and_Layor import *
import torch
import torch.nn as nn


class IAN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(IAN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        #self.lstm_context = Dynamic_LSTM(opt.embedding_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_context_R = Dynamic_LSTM(opt.embedding_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_context_L = Dynamic_LSTM(opt.embedding_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = Dynamic_LSTM(opt.embedding_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')
        #self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context_R = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context_L = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim*2, opt.num_class)

    def forward(self, inputs):
        #text_raw_indices, aspect_indices = inputs['context'], inputs['context']
        text_raw_indices_R, text_raw_indices_L, aspect_indices = inputs['context'], inputs['context'], inputs['context']
        #text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        text_raw_len_R = torch.sum(text_raw_indices_R != 0, dim=-1)
        text_raw_len_L = torch.sum(text_raw_indices_L != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        #context = self.embed(text_raw_indices)
        context_R = self.embed(text_raw_indices_R)
        context_L = self.embed(text_raw_indices_L)
        aspect = self.embed(aspect_indices)
        #context, (_, _) = self.lstm_context(context, text_raw_len)
        context_R, (_, _) = self.lstm_context_R(context_R, text_raw_len_R)
        context_L, (_, _) = self.lstm_context_L(context_L, text_raw_len_L)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        #text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.opt.device)
        #context_pool = torch.sum(context, dim=1)
        #context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        text_raw_len_R = torch.tensor(text_raw_len_R, dtype=torch.float).to(self.opt.device)
        context_pool_R = torch.sum(context_R, dim=1)
        context_pool_R = torch.div(context_pool_R, text_raw_len_R.view(text_raw_len_R.size(0), 1))

        text_raw_len_L = torch.tensor(text_raw_len_L, dtype=torch.float).to(self.opt.device)
        context_pool_L = torch.sum(context_L, dim=1)
        context_pool_L = torch.div(context_pool_L, text_raw_len_L.view(text_raw_len_L.size(0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool_R, context_pool_L)
        aspect_final = aspect_final.squeeze(dim=1)
        #context_final, _ = self.attention_context(context, aspect_pool)
        context_final_R, _ = self.attention_context_R(context_R, aspect_pool)
        #context_final = context_final.squeeze(dim=1)
        context_final_R = context_final_R.squeeze(dim=1)

        context_final_L, _ = self.attention_context_L(context_L, aspect_pool)
        context_final_L = context_final_L.squeeze(dim=1)

        #x = torch.cat((aspect_final, context_final), dim=-1)
        x = torch.cat((aspect_final, context_final_R, context_final_L), dim=-1)
        out = self.dense(x)
        return out
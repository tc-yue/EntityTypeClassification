# -*- coding: utf-8 -*-
# @Time    : 2019/6/16 0:56
# @Author  : Tianchiyue
# @File    : layers.py
# @Software: PyCharm
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim, embed_dim, mode='self'):
        super(Attention, self).__init__()
        self.mode = mode
        if self.mode == 'self':
            self.linear1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.tanh = nn.Tanh()
            self.linear2 = nn.Linear(hidden_dim, 1, bias=False)
        elif self.mode == 'bilinear':
            self.linear = nn.Linear(hidden_dim, embed_dim, bias=False)
        else:
            self.linear = nn.Linear(hidden_dim + embed_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sentence_rep, mask, query=None):
        if query is None:
            x1 = self.linear1(sentence_rep)
            x1 = self.tanh(x1)
            query_score = self.linear2(x1)
        if self.mode == 'bilinear':
            query = query.unsqueeze(2)
            sentence_linear = self.linear(sentence_rep)
            query_score = torch.bmm(sentence_linear, query)
        elif self.mode == 'concat':
            query = query.unsqueeze(1).expand(sentence_rep.size(0), sentence_rep.size(1),
                                              -1)  # bsz, seq_len,embed_dim
            query_sentence = torch.cat([query, sentence_rep], dim=-1)
            query_score = self.linear(query_sentence)
        query_score = query_score.squeeze(2)
        extend_attention_mask = (1.0 - mask) * -10000.0
        attention_score = query_score + extend_attention_mask  # bsz,seq_len
        attention_score = self.softmax(attention_score)
        weighted_rep = torch.bmm(attention_score.unsqueeze(1), sentence_rep).squeeze(1)
        return weighted_rep, attention_score


class DynamicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
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
        super(DynamicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""

        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            time_steps = x.size(1)
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first,
                                                         total_length=time_steps)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)
            return out, (ht, ct)

# -*- coding: utf-8 -*-
# @Time    : 2019/6/16 22:26
# @Author  : Tianchiyue
# @File    : model.py
# @Software: PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import DynamicRNN, Attention


class AttentiveLSTM(nn.Module):
    def __init__(self, args, word_emb_matrix=None):
        super(AttentiveLSTM, self).__init__()
        self.args = args

        if word_emb_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(word_emb_matrix, dtype=torch.float))
            self.embedding.weight.requires_grad = args.trainable_embedding
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
            self.embedding.weight.requires_grad = True

        self.lstm_l = DynamicRNN(args.embedding_dim, args.hidden_dim, num_layers=1, batch_first=True,
                                 bidirectional=True)
        self.lstm_r = DynamicRNN(args.embedding_dim, args.hidden_dim, num_layers=1, batch_first=True,
                                 bidirectional=True)

        self.attention = Attention(args.hidden_dim * 2, args.embedding_dim, mode=args.attention_mode)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(args.hidden_dim * 2 + args.embedding_dim, args.num_labels, bias=False)
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid_fn = nn.Sigmoid()

    def forward(self, inputs, labels=None):
        x_l, x_entity, x_r = inputs[0], inputs[1], inputs[2]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_entity_len = torch.tensor(torch.sum(x_entity != 0, dim=-1), dtype=torch.float32).cuda()

        x_l_mask = x_l.gt(0)
        x_r_mask = x_r.gt(0)

        x_l, x_r = self.embedding(x_l), self.embedding(x_r)
        entity_rep = self.embedding(x_entity)
        entity_rep = torch.div(torch.sum(entity_rep, dim=1), x_entity_len.view(-1, 1))

        left_outputs, _ = self.lstm_l(x_l, x_l_len)
        right_outputs, _ = self.lstm_r(x_r, x_r_len)
        #         left_outputs, _ = self.left_lstm(x_l)
        #         right_outputs,_ = self.right_lstm(x_r)
        context_rep = torch.cat((left_outputs, right_outputs), 1)
        #         context_rep = torch.max(left_outputs,1)[0]
        attention_mask = torch.cat((x_l_mask[:, :left_outputs.size(1)], x_r_mask[:, :right_outputs.size(1)]),
                                   dim=-1).to(dtype=torch.float32)
        context_rep, _ = self.attention(context_rep, attention_mask, entity_rep)
        all_rep = torch.cat((context_rep, entity_rep), dim=-1)
        all_rep = self.dropout(all_rep)
        logits = self.classifier(all_rep)
        if labels is not None:
            loss = self.loss(logits, labels)
            return loss
        else:

            return logits

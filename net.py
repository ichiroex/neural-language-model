# -*- coding: utf-8 -*-
"""
Network Architecture of Neural Language Model
"""
from __future__ import print_function
import chainer
from chainer import cuda, Variable, Chain
import chainer.functions as F
import chainer.links as L
import numpy as np

class NLM(Chain):

    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 context_window):

        super(NLM, self).__init__(
            embed = L.EmbedID(vocab_size, embed_size),
            l1 = L.Linear(embed_size * (context_window-1), hidden_size),
            l2 = L.Linear(hidden_size, vocab_size))

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.context_window = context_window

    def __call__(self, x1, x2):
        e1 = self.embed(x1)
        e2 = self.embed(x2)
        e = F.concat((e1, e2))
        h1 = F.tanh(self.l1(e))
        y = self.l2(h1)
        return y

    def get_embedding(self, x):
        return self.embed(x)

    def save_spec(self, filename):
        with open(filename, 'w') as fp:
            # パラメータを保存
            print(self.vocab_size, file=fp)
            print(self.embed_size, file=fp)
            print(self.hidden_size, file=fp)
            print(self.context_window, file=fp)

    @staticmethod
    def load_spec(filename):
        with open(filename) as fp:
            # specファイルからモデルのパラメータをロード
            vocab_size = int(next(fp))
            embed_size = int(next(fp))
            hidden_size = int(next(fp))
            context_window = int(next(fp))
            return NLM(vocab_size, embed_size, hidden_size, context_window)

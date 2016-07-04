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
                 hidden_size):

        super(NLM, self).__init__(
            embed = L.EmbedID(vocab_size, embed_size),
            l1 = L.Linear(embed_size*2, hidden_size),
            l2 = L.Linear(hidden_size, vocab_size))

    def __call__(self, x1, x2):
        e1 = self.embed(x1)
        e2 = self.embed(x2)
        e = F.concat((e1, e2))
        h1 = F.tanh(self.l1(e))
        y = self.l2(h1)
        return y

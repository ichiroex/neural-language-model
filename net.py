# -*- coding: utf-8 -*-
"""
Encoder-Decoderモデルをまとめて管理
"""
from __future__ import print_function
import chainer
from chainer import cuda, Variable, Chain
import chainer.functions as F
import chainer.links as L
import numpy as np

class CNNEncoder(Chain):
    # 数値用のEncoder

    def __init__(self,
                 input_channel,
                 output_channel,
                 filter_height,
                 filter_width,
                 mid_units,
                 hidden_size):
        super(CNNEncoder, self).__init__(
            conv1 = L.Convolution2D(input_channel,
                                    output_channel,
                                    (filter_height,
                                    filter_width)),
            l1    = L.Linear(mid_units, hidden_size),
            l2    = L.Linear(hidden_size, hidden_size)
        )

    def __call__(self, x, train=True):
        h = F.tanh(self.conv1(x))
        h = F.tanh(self.l1(h))
        h = self.l2(h)
        return h

class RNNEncoder(Chain):
    # シンボル用のEncoder
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(RNNEncoder, self).__init__(
        xe = L.EmbedID(vocab_size, embed_size),
        eh = L.Linear(embed_size, 4 * hidden_size),
        hh = L.Linear(hidden_size, 4 * hidden_size),
    )

  def __call__(self, x, c, h):
      e = F.tanh(self.xe(x))
      return F.lstm(c, self.eh(e) + self.hh(h))

class LSTMDecoder(Chain):

    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTMDecoder, self).__init__(
            ye = L.EmbedID(vocab_size, embed_size),
            eh = L.Linear(embed_size, 4 * hidden_size),
            hh = L.Linear(hidden_size, 4 * hidden_size),
            hf = L.Linear(hidden_size, embed_size),
            fy = L.Linear(embed_size, vocab_size),
        )

    def __call__(self, y, c, h):
        e    = F.tanh(self.ye(y))
        c, h = F.lstm(c, self.eh(e) + self.hh(h))
        f    = F.tanh(self.hf(h))
        y    = self.fy(f)
        return y, c, h

class EncoderDecoder(Chain):

    def __init__(self,
                input_channel,
                output_channel,
                filter_height,
                filter_width,
                mid_units,
                vocab_size,
                embed_size,
                hidden_size,
                gpu_flag):
        super(EncoderDecoder, self).__init__(
            cnnenc = CNNEncoder(input_channel,
                                output_channel,
                                filter_height,
                                filter_width,
                                mid_units,
                                hidden_size),
            rnnenc = RNNEncoder(vocab_size,
                                mid_units,
                                hidden_size),
            dec = LSTMDecoder(vocab_size,
                            embed_size,
                            hidden_size),
        )

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.mid_units = mid_units
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.myxp = cuda.cupy if gpu_flag else np

    def reset(self, batch_size):
        self.zerograds()
        self.c = Variable(self.myxp.zeros((batch_size,
                                           self.hidden_size),
                                           dtype=self.myxp.float32))
        self.h = Variable(self.myxp.zeros((batch_size,
                                           self.hidden_size),
                                           dtype=self.myxp.float32))
        """
        self.c = Variable(self.myxp.zeros((batch_size,
                                           self.hidden_size * 2),
                                           dtype=self.myxp.float32))
        self.cnn_c = Variable(self.myxp.zeros((batch_size,
                                           self.hidden_size),
                                           dtype=self.myxp.float32))
        self.rnn_c = Variable(self.myxp.zeros((batch_size,
                                           self.hidden_size),
                                           dtype=self.myxp.float32))
        self.h = Variable(self.myxp.zeros((batch_size,
                                           self.hidden_size * 2),
                                           dtype=self.myxp.float32))
        self.cnn_h = Variable(self.myxp.zeros((batch_size,
                                           self.hidden_size),
                                           dtype=self.myxp.float32))
        self.rnn_h = Variable(self.myxp.zeros((batch_size,
                                           self.hidden_size),
                                           dtype=self.myxp.float32))
        """

    def cnn_encode(self, x, train=True):
        self.h = self.cnnenc(x, train)

    def rnn_encode(self, x):
        self.rnn_c, self.rnn_h = self.rnnenc(x, self.c, self.h)

    def concat_state(self):
        self.h = F.concat((self.cnn_h, self.rnn_h))
        self.c = F.concat((self.cnn_c, self.rnn_c))

    def decode(self, y):
        y, self.c, self.h = self.dec(y, self.c, self.h)
        return y

    def save_spec(self, filename):
        with open(filename, 'w') as fp:
            # パラメータを保存
            print(self.input_channel, file=fp)
            print(self.output_channel, file=fp)
            print(self.filter_height, file=fp)
            print(self.filter_width, file=fp)
            print(self.mid_units, file=fp)
            print(self.vocab_size, file=fp)
            print(self.embed_size, file=fp)
            print(self.hidden_size, file=fp)


    @staticmethod
    def load_spec(filename, gpu_flag):
        with open(filename) as fp:
            # specファイルからモデルのパラメータをロード
            input_channel = int(next(fp))
            output_channel = int(next(fp))
            filter_height = int(next(fp))
            filter_width = int(next(fp))
            mid_units = int(next(fp))
            vocab_size = int(next(fp))
            embed_size = int(next(fp))
            hidden_size = int(next(fp))
            return EncoderDecoder(input_channel, output_channel, filter_height, filter_width, mid_units, vocab_size, embed_size, hidden_size, gpu_flag)

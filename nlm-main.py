# coding: utf-8
import numpy as np
import codecs
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import six
import sys
import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers, Variable
import chainer.functions as F
import argparse
from gensim import corpora, matutils
from gensim.models import word2vec
import time
import math
from net import EncoderDecoder
import mytools
import nltk.translate.bleu_score

"""
    Code for Neural Machine Translation
    Encoder-Decoder モデル

    Encoder: Convolutional Neural Networks
    Decoder: Reccurent Neural Networks
"""

def save_model(model, model_name):
    """ modelを保存
    """
    print ('save the model')
    serializers.save_npz('./models/' + model_name + '.model', model)

def save_optimizer(optimizer, model_name):
    """ optimzierを保存
    """
    print ('save the optimizer')
    serializers.save_npz('./models/' + model_name + '.state', optimizer)


def argument_parser():
    """ オプション設定
    """

    # デフォルト値の設定
    def_train = False
    def_test = False
    def_gpu = False
    def_is_debug_mode = False
    def_src = "./dataset/weather_src.train"
    def_trg = "./dataset/weather_trg.train"
    def_model = "encdec"

    # Encoder parameter
    def_input_channel = 1
    def_output_channel = 100
    def_filter_height = 11
    def_filter_width = 3
    def_mid_units = 500

    #Decoder parameter
    def_vocab = 1000
    def_embed = 100
    def_hidden = 200

    # Other parameter
    def_epoch = 10
    def_batchsize = 40
    def_grad_clip = 5

    # 引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('src',
                        type=str,
                        default=def_src,
                        help='source file')
    parser.add_argument('trg',
                        type=str,
                        default=def_trg,
                        help='target file')
    parser.add_argument('--train',
                        dest="train",
                        action="store_true",
                        default=def_train,
                        help="if set, run train mode")
    parser.add_argument('--test',
                        dest="test",
                        action="store_true",
                        default=def_test,
                        help="if set, run test mode")
    parser.add_argument('-d',
                        '--debug',
                        dest="is_debug_mode",
                        action="store_true",
                        default=def_is_debug_mode,
                        help="if set, run train with debug mode")
    parser.add_argument('--use-gpu  ',
                        dest='use_gpu',
                        action="store_true",
                        default=def_gpu,
                        help='use gpu')
    parser.add_argument('--model ',
                        dest='model',
                        type=str,
                        default=def_model,
                        help='model file name to save')
    parser.add_argument('--ich',
                        dest='input_channel',
                        type=int,
                        default=def_input_channel,
                        help='number of input channel')
    parser.add_argument('--och',
                        dest='output_channel',
                        type=int,
                        default=def_output_channel,
                        help='number of output channel')
    parser.add_argument('--fheight',
                        dest='filter_height',
                        type=int,
                        default=def_filter_height,
                        help='size of filter height')
    parser.add_argument('--fwidth',
                        dest='filter_width',
                        type=int,
                        default=def_filter_width,
                        help='size of filter width')
    parser.add_argument('--munits ',
                        dest='mid_units',
                        type=int,
                        default=def_mid_units,
                        help='size of mid units')
    parser.add_argument('--vocab ',
                        dest='vocab',
                        type=int,
                        default=def_vocab,
                        help='vocabulary size')
    parser.add_argument('--embed',
                        dest='embed',
                        type=int,
                        default=def_embed,
                        help='embedding layer size')
    parser.add_argument('--hidden',
                        dest='hidden',
                        type=int,
                        default=def_hidden,
                        help='hidden layer size')

    parser.add_argument('--epoch',
                        dest='epoch',
                        type=int,
                        default=def_epoch,
                        help='number of epochs to learn')
    parser.add_argument('--batchsize',
                        dest='batchsize'  ,
                        type=int,
                        default=def_batchsize,
                        help='learning minibatch size')
    parser.add_argument('--gclip',
                        dest='grad_clip'  ,
                        type=int,
                        default=def_grad_clip,
                        help='threshold of gradiation clipping')

    return parser.parse_args()


def forward_one_step(model,
                     src_num_batch,
                     src_sym_batch,
                     trg_batch,
                     src_vocab2id,
                     trg_vocab2id,
                     is_train,
                     xp):
    """ 損失を計算
    """
    generation_limit = 256
    batch_size = len(src_num_batch)
    model.reset(batch_size) # 状態をリセット

    # CNN Encoder
    num_x = Variable(xp.asarray(src_num_batch, dtype=xp.float32))
    model.cnn_encode(num_x)
    """
    # LSTM-RNN Encoder
    sym_x = Variable(xp.asarray([src_vocab2id['</s>'] for _ in range(batch_size)], dtype=xp.int32))
    model.rnn_encode(sym_x)
    src_sym_batch = xp.asarray(src_sym_batch, dtype=xp.int32).T # 転置
    for batch in reversed(src_sym_batch):
        sym_x = Variable(batch) #source
        model.rnn_encode(sym_x)


    model.concat_state()
    """
    # LSTM-RNN Decoder
    #print "Decoder..."
    t = Variable(xp.asarray([trg_vocab2id['<s>'] for _ in range(batch_size)], dtype=xp.int32))
    hyp_batch = [[] for _ in range(batch_size)]

    #print t.data
    # Train
    if is_train:

        loss = Variable(xp.asarray(xp.zeros(()), dtype=xp.float32))
        trg_batch = xp.asarray(trg_batch, dtype=xp.int32).T # 転置

        for batch in trg_batch:
            y = model.decode(t)
            t = Variable(batch) #target
            #print t.data
            loss += F.softmax_cross_entropy(y, t)
            output = cuda.to_cpu(y.data.argmax(1))
            for k in range(batch_size):
                hyp_batch[k].append(output[k])

        return hyp_batch, loss # 予測結果と損失を返す

    # Test
    else:
        while len(hyp_batch[0]) < generation_limit:
            y = model.decode(t)
            output = cuda.to_cpu(y.data.argmax(1))
            t = Variable(xp.asarray(output, dtype=xp.int32))

            for k in range(batch_size):
                hyp_batch[k].append(output[k])
            if all(hyp_batch[k][-1] == trg_vocab2id['</s>'] for k in range(batch_size)):
                break
        return hyp_batch # 予測結果を返す


def train(args):
    """ 学習を行うメソッド
    """

    # オプションの値をメソッド内の変数に渡す
    input_channel = args.input_channel   # inputチャネル数
    output_channel = args.output_channel # outputチャネル数
    filter_height = args.filter_height # filterの高さ
    filter_width = args.filter_width   # filterの幅
    mid_units = args.mid_units    # 中間層のユニット数
    vocab_size  = args.vocab      # 語彙数
    embed_size  = args.embed      # embeddingの次元数
    hidden_size = args.hidden     # 隠れ層のユニット数
    batchsize   = args.batchsize  # バッチサイズ
    n_epoch     = args.epoch      # エポック数(パラメータ更新回数)
    grad_clip   = args.grad_clip  # gradiation clip

    # 学習データの読み込み
    # Source
    print 'loading source text...'
    src_num_dataset, src_sym_dataset, src_vocab2id, src_id2vocab = mytools.load_src_data(args.src, vocab_size)

    # 正規化前のSourceファイル
    src_raw_num_dataset, src_raw_sym_dataset, _, _ = mytools.load_src_data(args.src + ".raw", vocab_size)

    # Target
    print 'loading target text...'
    trg_dataset, trg_vocab2id, trg_id2vocab = mytools.load_trg_data(args.trg, vocab_size)

    # 数値データセットの次元 (サンプル数, 数値ベクトルの種類数, 数値ベクトルの次元)
    sample_size, height, width = src_num_dataset.shape

    # CNNの入力用に変換
    # (sample size, # of channel, height, width) の4次元テンソルに変換
    src_num_dataset = src_num_dataset.reshape(sample_size,
                                              input_channel,
                                              height,
                                              width)

    # debug modeの時, パラメータの確認
    if args.is_debug_mode:
        print "[PARAMETERS]"
        print 'input channel:', input_channel
        print 'output channel:', output_channel
        print 'filter height:', filter_height
        print 'filter width:', filter_width
        print 'mid units:', mid_units

        print 'vocab size:', vocab_size
        print 'embed size:', embed_size
        print 'hidden size:', hidden_size
        print 'mini batch size:', batchsize
        print 'epoch:', n_epoch
        print 'grad clip threshold:', grad_clip
        print
        print 'sample size:', sample_size
        print 'height:', height
        print 'width:', width
        print

    # モデルの定義
    model = EncoderDecoder(input_channel,
                           output_channel,
                           filter_height,
                           filter_width,
                           mid_units,
                           vocab_size,   # 語彙数
                           embed_size,   # embeddinの次元数
                           hidden_size,  # 隠れ層のユニット数
                           args.use_gpu) # GPUを使うかどうか

    # GPUを使うかどうか
    if args.use_gpu:
        cuda.check_cuda_available()
        cuda.get_device(1).use()
        model.to_gpu()
    xp = cuda.cupy if args.use_gpu else np #args.gpu <= 0: use cpu, otherwise: use gpu


    N = sample_size
    # Setup optimizer
    optimizer = optimizers.AdaGrad(lr=0.01)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

    # 学習の始まり
    for epoch in range(n_epoch):
        print 'epoch:', epoch, '/', n_epoch

        # training
        perm = np.random.permutation(N) #ランダムな整数列リストを取得
        sum_train_loss = 0.0

        j = 0
        for i in six.moves.range(0, N, batchsize):

            #perm を使い x_train, y_trainからデータセットを選択 (毎回対象となるデータは異なる)
            src_num_batch = src_num_dataset[perm[i:i + batchsize]]
            src_sym_batch = src_sym_dataset[perm[i:i + batchsize]]

            # 出力用のソースバッチ
            src_raw_num_batch = src_raw_num_dataset[perm[i:i + batchsize]]
            src_raw_sym_batch = src_raw_sym_dataset[perm[i:i + batchsize]]

            # Target batch
            trg_batch = trg_dataset[perm[i:i + batchsize]]

            # 各バッチ内のサイズを統一させる
            src_sym_batch = mytools.fill_batch(src_sym_batch, src_vocab2id['</s>'])
            trg_batch = mytools.fill_batch(trg_batch, trg_vocab2id['</s>'])

            # 損失を計算
            hyp_batch, loss = forward_one_step(model,
                                               src_num_batch,
                                               src_sym_batch,
                                               trg_batch,
                                               src_vocab2id,
                                               trg_vocab2id,
                                               args.train,
                                               xp) # is_train

            sum_train_loss  += float(cuda.to_cpu(loss.data)) * len(trg_batch)   # 平均誤差計算用

            loss.backward() # Backpropagation
            optimizer.update() # 重みを更新

            # 学習の途中経過を表示
            for k in range(len(src_num_batch)):
                # 辞書を利用して単語列を出力
                print 'epoch: ', epoch, ', sample:', batchsize * j + k
                #print 'src_num:', src_num_batch[k]
                #print 'src_sym:', " ".join( [ src_id2vocab[x] for x in src_sym_batch[k] ])

                print 'raw_src_num:', src_raw_num_batch[k]
                print 'raw_src_sym:', " ".join( [ src_id2vocab[x] for x in src_raw_sym_batch[k] ])
                print 'trg:', ' '.join( [trg_id2vocab[x] for x in trg_batch[k]] )
                print 'hyp:', ' '.join( [trg_id2vocab[x] for x in hyp_batch[k]] )
                print "================="

            j += 1

        print('train mean loss={}'.format(sum_train_loss / N)) #平均誤差

        #モデルの途中経過を保存
        print 'saving model....'
        prefix = './model/' + args.model + '.%03.d' % (epoch + 1)
        mytools.save_vocab(prefix + '.srcvocab', src_id2vocab)
        mytools.save_vocab(prefix + '.trgvocab', trg_id2vocab)
        model.save_spec(prefix + '.spec')
        serializers.save_hdf5(prefix + '.weights', model)

        sys.stdout.flush()

def test(args):
    """ 予測を行うメソッド
    """

    input_channel = args.input_channel # 入力チャネル
    batchsize   = args.batchsize  # バッチサイズ

    # 語彙辞書の読込
    src_vocab2id, src_id2vocab, vocab_size = mytools.load_vocab(args.model + ".srcvocab")
    trg_vocab2id, trg_id2vocab, vocab_size = mytools.load_vocab(args.model + ".trgvocab")

    # モデルの読込
    model = EncoderDecoder.load_spec(args.model + ".spec", args.use_gpu)

    # GPUを使うかどうか
    if args.use_gpu:
        cuda.check_cuda_available()
        cuda.get_device(1).use()
        model.to_gpu()

    xp = cuda.cupy if args.use_gpu else np # args.gpu <= 0: use cpu, otherwise: use gpu
    serializers.load_hdf5(args.model + ".weights", model)

    # Source sequence for test
    print 'loading source numeric and symbolic data for test...'
    # 数値データ, シンボルデータ
    test_src_num_dataset, test_src_sym_dataset = mytools.load_test_src_data(args.src, src_vocab2id)

    # 正規化前のsrcファイル(出力用に読込)
    test_raw_src_num_dataset, test_raw_src_sym_dataset = mytools.load_test_src_data(args.src + ".raw", src_vocab2id)

    # テスト時の正解データ
    test_trg_dataset = mytools.load_test_trg_data(args.trg)

    # 数値データセットの次元 (サンプル数, 数値ベクトルの種類数, 数値ベクトルの次元)
    sample_size, height, width = test_src_num_dataset.shape

    # CNNの入力用に変換
    # (sample size, # of channel, height, width) の4次元テンソルに変換
    test_src_num_dataset = test_src_num_dataset.reshape(sample_size,
                                                      input_channel,
                                                      height,
                                                      width)

    print 'generating sequences ...'
    generated = 0
    N = len(test_src_num_dataset) # テストの事例数
    weights = [0.25, 0.25, 0.25, 0.25]
    bleu_score_list = []

    for i in six.moves.range(0, N, batchsize):

        # ミニバッチの作成
        # Source
        src_num_batch = test_src_num_dataset[i:i+batchsize]
        src_sym_batch = test_src_sym_dataset[i:i+batchsize]

        # 出力用のSource batch
        raw_src_num_batch = test_raw_src_num_dataset[i:i+batchsize]
        raw_src_sym_batch = test_raw_src_sym_dataset[i:i+batchsize]

        # Target
        trg_batch = test_trg_dataset[i:i+batchsize]

        # 各バッチのサイズを統一させる
        src_sym_batch = mytools.fill_batch(src_sym_batch, src_vocab2id['</s>'])

        K = len(src_num_batch)
        print 'sample %8d - %8d ...' % (generated + 1, generated + K)

        # 損失を計算
        hyp_batch = forward_one_step(model,
                                   src_num_batch,
                                   src_sym_batch,
                                   None,
                                   src_vocab2id,
                                   trg_vocab2id,
                                   False,
                                   xp) # is_train

        k = 0
        for hyp in hyp_batch:
            hyp.append('</s>')
            hyp = hyp[:hyp.index('</s>')]
            #print 'src_num:', src_num_batch[k]
            #print 'src_sym:', ' '.join([ src_id2vocab[x] for li in src_sym_batch[k] for x in li ])

            print 'raw_src_num:', raw_src_num_batch[k]
            print 'raw_src_sym:', ' '.join([ src_id2vocab[x] for x in raw_src_sym_batch[k] ])
            _trg = [x for x in trg_batch[k]]
            _hyp = [trg_id2vocab[x] if trg_id2vocab[x] != "</s>" else "" for x in hyp]
            bleu_score = nltk.translate.bleu([_trg], _hyp, weights)
            bleu_score_list.append(bleu_score)
            print 'trg:', ''.join( _trg )
            print 'hyp:', ''.join( _hyp )
            print 'BLEU:', bleu_score
            print '=============================================='
            k += 1

        generated += K

        sys.stdout.flush()

    print 'BLEU:', np.mean(np.array(bleu_score_list))
    print 'finished.'


def main():
    args = argument_parser()

    if args.train:
        train(args)
    elif args.test:
        test(args)


if __name__ == "__main__":
    main()
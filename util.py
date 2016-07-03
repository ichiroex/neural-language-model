# -*- coding: utf-8 -*-

import sys
import numpy as np
from collections import defaultdict
from scipy import stats

"""
自分で作成したTOOLとか
"""

def str_isfloat(str):
    """
    strが数値化どうか判定
    """
    try:
        float(str)
        return True

    except ValueError:
        return False

def replace_unknown_words(lines, N):
    """
        出現回数がN未満の単語を未知語として<unk>に置換する.
    """

    vocab = {}
    sentences = []
    output = []

    # 各単語(key)の出現回数をvalueとする辞書を作成
    for line in lines:
        for w in line.split():
            vocab[w] = vocab.get(w, 0) + 1
        sentences.append(line.split())


    # 頻度N以下の単語は未知語<unk>とする
    for sen in sentences:
        tmp = []
        for word in sen:
            if vocab[word] < N:
                tmp.append('<unk>')
            else:
                tmp.append(word)
        output.append(tmp)
    return output

def standardization(numeric_data_list):
    """
    数値データを同系の値ごとに標準化を行う (平均:0, 分散:1)
    """

    # 転置
    t_numeric_data_list = np.array(numeric_data_list).T

    numeric_data_list = []
    for val_list in t_numeric_data_list:

        data_list = np.array([ num for li in val_list for num in li if num != None ]) # 同じ種類のデータリスト

        zscore = stats.zscore(data_list) #　数値データを標準化

        # Noneの部分に0を入れて元の形に戻す
        i = 0
        nval_list = []
        for li in val_list:
            tmp = []
            for num in li:
                if num != None:
                    tmp.append(zscore[i])
                    i += 1
                else:
                    tmp.append(0)
            nval_list.append(tmp)

        numeric_data_list.append(nval_list)


    return np.array(numeric_data_list).T


def padding(data_list):

    """
    次元数がmax_len以下のrowは0でpadding
    """
    max_len = max ( len (data) for data in data_list )

    return [ data + [0] * (max_len - len(data)) for data in data_list]


# input data
def load_src_data(fname, vocab_size):

    """
    ソースファイルを読み込み
    数値データとシンボルデータを分けてデータを入力
    """

    print fname

    numeric_data_list = []  # 全事例の数値を格納
    symbolic_data_list = [] # 全事例のシンボルを格納
    symbol_freq = defaultdict(lambda: 0) # 各シンボルの出現回数計算用

    with open(fname, "r") as f:
        # ファイルを一行ずつ渡す
        for line in f:
            numeric_line, symbolic_line = line.split("|")

            numeric_data = [ [ float(n) if str_isfloat(n) else None for n in ne.split(",")] for ne in numeric_line.split("\t") ]
            symbolic_data = [ s.strip()  for se in symbolic_line.split("\t") for s in se.split(",") ]

            # 数値データをリストに格納
            numeric_data_list.append(numeric_data)

            # シンボルデータをリストに格納
            symbolic_data_list.append(symbolic_data)

            # 各シンボルの出現回数を数える
            for symbol in symbolic_data:
                symbol_freq[symbol] += 1

    # 数値データを標準化
    #numeric_data_list = standardization(numeric_data_list)

    # padding
    #numeric_dataset = [padding(data_list) for data_list in numeric_data_list]

    #標準化+padding済みのデータを読み込んだので、そのまま入れる
    numeric_dataset = numeric_data_list

    # 単語-ID、ID-単語 辞書を作成
    vocab2id = defaultdict(lambda: 0)
    vocab2id['<unk>'] = 0
    vocab2id['<s>'] = 1
    vocab2id['</s>'] = 2

    id2vocab = [""] * vocab_size
    id2vocab[0] = '<unk>'
    id2vocab[1] = '<s>'
    id2vocab[2] = '</s>'

    # シンボル辞書を作成
    for i, (symbol, count) in zip(range(vocab_size - 3), sorted(symbol_freq.items(), key=lambda x:-x[1])):
        vocab2id[symbol] += i + 3
        id2vocab[i + 3]  += symbol

    # id化したシンボルデータセット
    symbolic_dataset = [ [ vocab2id.get(symbol, vocab2id["<unk>"]) for symbol in symbolic_data ] for symbolic_data in symbolic_data_list ]

    print 'dataset size', len(symbolic_dataset)
    print 'symbol vocab size:', len(vocab2id)
    print 'symbol vocab size(actual):', len(symbol_freq)
    print

    return np.array(numeric_dataset), np.array(symbolic_dataset), vocab2id, id2vocab

# input data
def load_trg_data(fname, vocab_size):

    """
    ターゲットファイルを読み込み、データセット, 単語辞書を返す
    語彙数を指定する
    """

    print fname
    lines = open(fname, 'r').readlines()

    # 単語リストと文頭・文末文字を追加した文書リストを作成
    words = []
    doc   = []
    word_freq = defaultdict(lambda: 0)
    for line in lines:
        sentence = line.strip().split()
        doc.append(sentence)
        words.extend(sentence)

        # 単語の出現回数を計算
        for word in sentence:
            word_freq[word] += 1

    # 単語-ID、ID-単語 辞書を作成
    vocab2id = defaultdict(lambda: 0)
    vocab2id['<unk>'] = 0
    vocab2id['<s>'] = 1
    vocab2id['</s>'] = 2

    id2vocab = [""] * vocab_size
    id2vocab[0] = '<unk>'
    id2vocab[1] = '<s>'
    id2vocab[2] = '</s>'

    for i, (word, count) in zip(range(vocab_size - 3), sorted(word_freq.items(), key=lambda x:-x[1])):
        vocab2id[word] = i + 3
        id2vocab[i + 3] = word

    dataset = [[ vocab2id.get(word, 0) for word in sen ] for sen in doc] # 各事例をindex系列に置き換える

    print 'dataset size', len(dataset)
    print 'corpus length:', len(words)
    print 'vocab size:', len(vocab2id)
    print 'vocab size(actual):', len(word_freq)
    print

    return np.array(dataset), vocab2id, id2vocab

# input data
def load_test_src_data(fname, vocab2id):

    """
    テストデータ用の読み込み用
    学習データのvocab2id辞書を基に、単語列をid列に変換
    """

    print fname

    numeric_data_list = []  # 全事例の数値を格納
    symbolic_data_list = [] # 全事例のシンボルを格納
    symbol_freq = defaultdict(lambda: 0) # 各シンボルの出現回数計算用

    with open(fname, "r") as f:
        # ファイルを一行ずつ渡す
        for line in f:
            numeric_line, symbolic_line = line.split("|")

            numeric_data = [ [ float(n) if str_isfloat(n) else None for n in ne.split(",")] for ne in numeric_line.split("\t") ]
            symbolic_data = [ s.strip() for se in symbolic_line.split("\t") for s in se.split(",") ]

            # 数値データをリストに格納
            numeric_data_list.append(numeric_data)

            # シンボルデータをリストに格納
            symbolic_data_list.append(symbolic_data)

    # 数値データセット
    numeric_dataset = numeric_data_list

    # id化したシンボルデータセット
    symbolic_dataset = [ [ vocab2id.get(symbol, vocab2id["<unk>"]) for symbol in symbolic_data ] for symbolic_data in symbolic_data_list ]

    print 'dataset size', len(symbolic_dataset)
    print

    return np.array(numeric_dataset), np.array(symbolic_dataset)

# input data
def load_test_trg_data(fname):

    """
    テストデータ用の読み込み用
    """

    print fname
    lines = open(fname, 'r').readlines()

    # 文書リストを作成
    dataset   = [ line.strip().split() for line in lines ]

    print 'dataset size', len(dataset)
    print

    return np.array(dataset)



# バッチ内の全てのベクトルを同じ次元に揃える
def fill_batch(batch, token_id):
    max_len = max(len(x) for x in batch)

    filled_batch = []
    for x in batch:
        if len(x) < max_len:
            padding = [ token_id for _ in range(max_len - len(x))]
            filled_batch.append(x + padding)
        else:
            filled_batch.append(x)
    return filled_batch

def save_vocab(filename, id2vocab):
    """ ファイルに語彙辞書を保存
    """
    with open(filename, 'w') as fp:
        print >> fp, len(id2vocab)
        for i in range(len(id2vocab)):
            print >> fp, id2vocab[i]

def load_vocab(filename):
    """ ファイルから語彙辞書を読込
    """
    with open(filename) as fp:
        vocab_size = int(next(fp))
        vocab2id = defaultdict(lambda: 0)
        id2vocab = [""] * vocab_size
        for i in range(vocab_size):
            s = next(fp).strip()
            if s:
                vocab2id[s] = i
                id2vocab[i] = s

    return vocab2id, id2vocab, vocab_size


if __name__ == '__main__':

    N = 5 # threshold of unknown word
    lines = open(sys.argv[1], 'r').readlines()
    output = replace_unknown_words(lines, N)

    for sen in output:
        print ' '.join(sen)

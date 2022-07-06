import sys
import logging
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
from h01_data.alphabet import Alphabet
from h01_data.filter_data import count_sentences
from util.argparser import get_argparser, parse_args, add_data_args
from util import util


def get_args():
    argparser = get_argparser()
    argparser.add_argument("--wikipedia-tokenized-file", type=str,)
    argparser.add_argument("--n-folds", type=int, default=10,)
    argparser.add_argument("--no-shuffle", action='store_true')
    add_data_args(argparser)
    return parse_args(argparser)


def get_fold_splits(n_sentences, n_folds, shuffle_splits=True):
    splits = np.arange(n_sentences)
    if shuffle_splits:
        np.random.shuffle(splits)
    splits = np.array_split(splits, n_folds)
    splits = {x: i for i, fold in enumerate(splits) for x in fold}
    return splits


def process_line(line, word_info, sentence_list, alphabet):
    sentence = line.strip().replace('-', ' ').split(' ')

    sentence_list.append(line.strip())
    for word in sentence:
        word = word.lower()
        alphabet.add_word(word)

        word_info[word] = word_info.get(word, 0) + 1


def process_data(src_fname, n_folds, splits, alphabet):
    word_folds = [{} for _ in range(n_folds)]
    sentence_folds = [[] for _ in range(n_folds)]
    with open(src_fname, 'r', encoding='utf8') as f:
        for i, line in tqdm(enumerate(f), desc='Processing wiki data',
                            total=len(splits)):
            fold = splits[i]
            process_line(line, word_folds[fold], sentence_folds[fold],
                         alphabet)
    return word_folds, sentence_folds


def count_tokens(folds):
    return [sum(list(word_info.values())) for word_info in folds]


def count_types(folds):
    return [len(word_info) for word_info in folds]


def process(src_fname, tgt_fname, n_folds, shuffle_splits=True):
    n_sentences = count_sentences(src_fname)
    splits = get_fold_splits(n_sentences, n_folds, shuffle_splits)
    alphabet = Alphabet()

    word_folds, sentence_folds = process_data(src_fname, n_folds, splits,
                                              alphabet)
    n_tokens = count_tokens(word_folds)
    n_types = count_types(word_folds)
    util.write_data(tgt_fname, (word_folds, alphabet, sentence_folds))

    print('# unique chars:', len(alphabet))
    print('# tokens per fold:', n_tokens)
    print('# types per fold:', n_types)


def main():
    args = get_args()
    logging.info(args)

    process(args.wikipedia_tokenized_file, args.data_file,
            args.n_folds, shuffle_splits=(not args.no_shuffle))


if __name__ == '__main__':
    main()

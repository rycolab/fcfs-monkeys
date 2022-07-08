import os
import sys
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import constants
from util import util
from util.argparser import get_argparser, parse_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--checkpoints-path', type=str, required=True)

    args = parse_args(argparser)
    util.config(args.seed)
    return args


def str_for_table(corr, pvalue):
    if pvalue < 0.01:
        pvalue_str = '$\\ddagger$'
    elif pvalue < 0.05:
        pvalue_str = '$\\dagger$'
    else:
        pvalue_str = ''
    return '%.2f%s' % (corr, pvalue_str)


def print_table1(df):
    table_str = '%s & %s & %s & %s \\\\'

    df_mean = df.groupby('language').agg('mean').reset_index()

    natural_pvalue = util.permutation_test(df.natural_frequency_corr.to_numpy())
    fcfs_pvalue = util.permutation_test(df.fcfs_frequency_corr.to_numpy())
    polyfcfs_pvalue = util.permutation_test(df.polyfcfs_frequency_corr.to_numpy())

    language = constants.LANG_NAMES[df_mean['language'].item()]
    natural_corr = str_for_table(
        df_mean['natural_frequency_corr'].item(), natural_pvalue)
    fcfs_corr = str_for_table(
        df_mean['fcfs_frequency_corr'].item(), fcfs_pvalue)
    polyfcfs_corr = str_for_table(
        df_mean['polyfcfs_frequency_corr'].item(), polyfcfs_pvalue)

    print(table_str % (language, natural_corr, fcfs_corr, polyfcfs_corr))


def print_table2(df):
    table_str = '%s & %d & %s & %s & %s \\\\'

    df_mean = df.groupby('language').agg('mean').reset_index()

    language = constants.LANG_NAMES[df_mean['language'].item()]
    if language in ['Hebrew', 'Turkish']:
        n_types_wn = df_mean['n_types_polysemy_natural'].item()
    else:
        n_types_wn = df_mean['n_types_wordnet'].item()

    wordnet_vs_polysemy_pvalue = util.permutation_test(df.wornet_vs_polysemy_corr.to_numpy()) \
        if not df.wornet_vs_polysemy_corr.isna().any() else float('nan')
    natural_wordnet_len_pvalue = util.permutation_test(df.natural_wordnet_len_corr.to_numpy()) \
        if not df.natural_wordnet_len_corr.isna().any() else float('nan')
    natural_polysemy_pvalue = util.permutation_test(df.natural_polysemy_corr.to_numpy()) \
        if not df.natural_polysemy_corr.isna().any() else float('nan')

    wornet_vs_polysemy_corr = str_for_table(
        df_mean['wornet_vs_polysemy_corr'].item(), wordnet_vs_polysemy_pvalue)
    natural_wordnet_len_corr = str_for_table(
        df_mean['natural_wordnet_len_corr'].item(), natural_wordnet_len_pvalue)
    natural_polysemy_corr = str_for_table(
        df_mean['natural_polysemy_corr'].item(), natural_polysemy_pvalue)

    print(table_str %
          (language, n_types_wn, natural_wordnet_len_corr,
           natural_polysemy_corr, wornet_vs_polysemy_corr))


def print_table3(df):
    table_str = '%s & %s & %s & %s \\\\'

    df_mean = df.groupby('language').agg('mean').reset_index()

    # corr_diff = (df.polyfcfs_polysemy_corr - df.natural_polysemy_corr).to_numpy()
    natural_pvalue = util.permutation_test(df.natural_polysemy_corr.to_numpy())
    fcfs_pvalue = util.permutation_test(df.fcfs_polysemy_corr.to_numpy())
    polyfcfs_pvalue = util.permutation_test(df.polyfcfs_polysemy_corr.to_numpy())

    language = constants.LANG_NAMES[df_mean['language'].item()]
    natural_polysemy_corr = str_for_table(
        df_mean['natural_polysemy_corr'].item(), natural_pvalue)
    fcfs_polysemy_corr = str_for_table(
        df_mean['fcfs_polysemy_corr'].item(), fcfs_pvalue)
    polyfcfs_polysemy_corr = str_for_table(
        df_mean['polyfcfs_polysemy_corr'].item(), polyfcfs_pvalue)

    print(table_str % (language, natural_polysemy_corr, fcfs_polysemy_corr, polyfcfs_polysemy_corr))


def get_language_results(language, checkpoints_path):
    dfs = []
    for seed in range(10):
        results_compiled_file = os.path.join(
            checkpoints_path, language, 'seed_%02d' % seed, 'compiled_results.tsv')
        df = pd.read_csv(results_compiled_file, sep='\t')
        df['seed'] = seed
        dfs += [df]

    return pd.concat(dfs)


def print_results(language, checkpoints_path):
    df = get_language_results(language, checkpoints_path)

    pvalue = util.permutation_test(
        (df.fcfs_frequency_corr - df.natural_frequency_corr).to_numpy())
    print('%s. FCFS vs Natural frequency--length: %.4f' % (language, pvalue))
    pvalue = util.permutation_test(
        (df.polyfcfs_polysemy_corr - df.natural_polysemy_corr).to_numpy())
    print('%s. PolyFCFS vs Natural polysemy--length: %.4f' % (language, pvalue))

    print_table1(df)
    print_table2(df)
    print_table3(df)
    print()


def main():
    args = get_args()

    for language in constants.LANGUAGES:
        print_results(language, args.checkpoints_path)


if __name__ == '__main__':
    main()

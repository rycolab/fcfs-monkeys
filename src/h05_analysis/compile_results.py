import os
import sys
import pandas as pd
from scipy import stats
from nltk.corpus import wordnet as wn

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util
from util.argparser import get_argparser, parse_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--language', type=str, required=True)
    argparser.add_argument('--results-freq-codes-file', type=str, required=True)
    argparser.add_argument('--results-ent-polyassign-file', type=str, required=True)
    argparser.add_argument('--results-ent-natural-file', type=str, required=True)
    argparser.add_argument('--results-compiled-file', type=str, required=True)

    args = parse_args(argparser)
    util.config(args.seed)
    return args


def str_for_table(df, min_count, var1='poly_cov', var2='len'):

    corr, pvalue = stats.spearmanr(
        df.loc[df['count'] >= min_count, var1],
        df.loc[df['count'] >= min_count, var2])

    if pvalue < 0.01:
        pvalue_str = '$\\ddagger$'
    elif pvalue < 0.05:
        pvalue_str = '$\\dagger$'
    else:
        pvalue_str = ''
    return '%.2f%s' % (corr, pvalue_str)


def get_frequency_length_correlations(df, df_poly):
    natural_frequency_correlations = stats.spearmanr(df['frequencies'],
                                                     df['natural_length'])
    fcfs_frequency_correlations = stats.spearmanr(df['frequencies'],
                                                  df['fcfs_length'])
    polyfcfs_frequency_correlations = stats.spearmanr(df_poly['frequencies'],
                                                      df_poly['fcfs_length'])
    caplan_frequency_correlations = stats.spearmanr(df['frequencies'],
                                                    df['caplan_length'])
    polycaplan_frequency_correlations = stats.spearmanr(df_poly['frequencies'],
                                                        df_poly['caplan_length'])

    return natural_frequency_correlations, fcfs_frequency_correlations, \
        polyfcfs_frequency_correlations, caplan_frequency_correlations, \
        polycaplan_frequency_correlations


def get_wordnet_synsets(df, lang_code):
    df['natural'] = df['natural'].apply(str)
    df['wn_synset'] = df['natural'].apply(
        lambda x: len(wn.synsets(x, lang=lang_code)) if lang_code else -1)
    return df


def filter_min_count(df_nat, df_polyassign, min_count):
    df_polyassign = df_polyassign[df_polyassign['frequencies'] >= min_count]
    df_nat = df_nat[df_nat['frequencies'] >= min_count]

    return df_nat, df_polyassign


def get_polysemy_length_correlations(df_nat, df_polyassign, wn_eval, polysemy_col='poly_cov'):
    # WordNet correlations
    wornet_vs_polysemy_corr = stats.spearmanr(wn_eval['wn_synset'], wn_eval[polysemy_col])
    natural_wordnet_len_corr = stats.spearmanr(wn_eval['wn_synset'], wn_eval['natural_length'])

    # Polysemy correlations
    natural_polysemy_corr = stats.spearmanr(df_nat[polysemy_col],
                                            df_nat['natural_length'])
    fcfs_polysemy_corr = stats.spearmanr(df_nat[polysemy_col],
                                         df_nat['fcfs_length'])
    polyfcfs_polysemy_corr = stats.spearmanr(df_polyassign[polysemy_col],
                                             df_polyassign['fcfs_length'])
    caplan_polysemy_corr = stats.spearmanr(df_nat[polysemy_col],
                                           df_nat['caplan_length'])
    polycaplan_polysemy_corr = stats.spearmanr(df_polyassign[polysemy_col],
                                               df_polyassign['caplan_length'])

    return wornet_vs_polysemy_corr, natural_wordnet_len_corr, \
        natural_polysemy_corr, fcfs_polysemy_corr, polyfcfs_polysemy_corr, \
        caplan_polysemy_corr, polycaplan_polysemy_corr


def main():
    # pylint: disable=too-many-locals
    args = get_args()

    lang_codes = {
        'en': 'eng',
        'fi': 'fin',
        'pt': 'por',
        'id': 'ind',
        'he': None,
        'tr': None,
        'simple': 'eng',
    }

    # Get natural and fcfs code lengths
    df_length = pd.read_csv(args.results_freq_codes_file, sep='\t')
    del df_length['Unnamed: 0']

    # Get polysemy results
    df_nat = pd.read_csv(args.results_ent_natural_file, sep='\t')
    df_polyassign = pd.read_csv(args.results_ent_polyassign_file, sep='\t')

    # Get frequency--length correlations
    natural_frequency_corr, fcfs_frequency_corr, polyfcfs_frequency_corr, \
        caplan_frequency_corr, polycaplan_frequency_corr = \
        get_frequency_length_correlations(df_length, df_polyassign)

    # Filter words with less than 10 ocurrances
    df_nat, df_polyassign = \
        filter_min_count(df_nat, df_polyassign, min_count=10)

    # Get number of sense in Wordnet
    df_nat = get_wordnet_synsets(df_nat, lang_codes[args.language])
    wn_eval = df_nat[df_nat['wn_synset'] > 1]

    # Get polysemy--length correlations
    wornet_vs_polysemy_corr, natural_wordnet_len_corr, natural_polysemy_corr, \
        fcfs_polysemy_corr, polyfcfs_polysemy_corr, \
        caplan_polysemy_corr, polycaplan_polysemy_corr = \
        get_polysemy_length_correlations(df_nat, df_polyassign, wn_eval)

    results = {
        'language': [args.language],
        'n_types_frequency_experiment': df_length.shape[0],
        'n_types_wordnet': wn_eval.shape[0],
        'n_types_polysemy_natural': df_nat.shape[0],
        'n_types_polysemy_polyassign': df_polyassign.shape[0],
        'natural_frequency_corr': [natural_frequency_corr.correlation],
        'natural_frequency_corr--pvalue': [natural_frequency_corr.pvalue],
        'fcfs_frequency_corr': [fcfs_frequency_corr.correlation],
        'fcfs_frequency_corr--pvalue': [fcfs_frequency_corr.pvalue],
        'polyfcfs_frequency_corr': [polyfcfs_frequency_corr.correlation],
        'polyfcfs_frequency_corr--pvalue': [polyfcfs_frequency_corr.pvalue],
        'caplan_frequency_corr': [caplan_frequency_corr.correlation],
        'caplan_frequency_corr--pvalue': [caplan_frequency_corr.pvalue],
        'polycaplan_frequency_corr': [polycaplan_frequency_corr.correlation],
        'polycaplan_frequency_corr--pvalue': [polycaplan_frequency_corr.pvalue],
        'wornet_vs_polysemy_corr': [wornet_vs_polysemy_corr.correlation],
        'wornet_vs_polysemy_corr--pvalue': [wornet_vs_polysemy_corr.pvalue],
        'natural_wordnet_len_corr': [natural_wordnet_len_corr.correlation],
        'natural_wordnet_len_corr--pvalue': [natural_wordnet_len_corr.pvalue],
        'natural_polysemy_corr': [natural_polysemy_corr.correlation],
        'natural_polysemy_corr--pvalue': [natural_polysemy_corr.pvalue],
        'fcfs_polysemy_corr': [fcfs_polysemy_corr.correlation],
        'fcfs_polysemy_corr--pvalue': [fcfs_polysemy_corr.pvalue],
        'polyfcfs_polysemy_corr': [polyfcfs_polysemy_corr.correlation],
        'polyfcfs_polysemy_corr--pvalue': [polyfcfs_polysemy_corr.pvalue],
        'caplan_polysemy_corr': [caplan_polysemy_corr.correlation],
        'caplan_polysemy_corr--pvalue': [caplan_polysemy_corr.pvalue],
        'polycaplan_polysemy_corr': [polycaplan_polysemy_corr.correlation],
        'polycaplan_polysemy_corr--pvalue': [polycaplan_polysemy_corr.pvalue],
    }
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.results_compiled_file, sep='\t', index=False)


if __name__ == '__main__':
    main()

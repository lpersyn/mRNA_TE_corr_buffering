import argparse
import os
from train import run_experiment, RANDOM_SEED, get_model_dir_name, create_symbol_to_fold
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
from symbol_to_fold_with_class_balance import add_symbol_to_fold_with_class_balance
from data import get_data

def human_corr(args):
    targets_path = 'human_RNA_TE_corr.csv'
    symbol_to_fold_path = 'human_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = [
        'LL', 'P5', 'P3', 'CF', 'AAF',
        '3mer_freq_5'
    ]
    # feature_set = [
    #     'LL', 'P5', 'P3', 'CF', 'AAF',
    #     '3mer_freq_5', 'Struct', 'Biochem'
    # ]

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/human/RNA_TE_corr/',
        bio_source='bio_source_TE_RNA_cor_value_nond',
        species='human',
        )    
    
def human_corr_all(args):
    targets_path = 'human_RNA_TE_corr.csv'
    symbol_to_fold_path = 'human_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = ['LL', 'P', 'P5', 'PC', 'P3', 'WP', 'K', 'CF', 'AAF', 'DC', 
                   '1mer_freq_5', '1mer_freq_C', '1mer_freq_3', '2mer_freq_5', '2mer_freq_C', '2mer_freq_3', 
                   '3mer_freq_5', '3mer_freq_C', '3mer_freq_3', '4mer_freq_5', '4mer_freq_C', '4mer_freq_3', 
                   '5mer_freq_5', '5mer_freq_C', '5mer_freq_3', '6mer_freq_5', '6mer_freq_C', '6mer_freq_3', 
                   'Struct']

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/human/RNA_TE_corr/',
        bio_source='bio_source_TE_RNA_cor_value_nond',
        species='human',
        )   

def human_te(args):
    targets_path = 'human_TE.csv'
    symbol_to_fold_path = 'human_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = [
        'LL', 'P5', 'P3', 'CF', 'AAF',
        '3mer_freq_5'
    ]

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/human/TE/',
        bio_source='mean_te',
        species='human',
        )   

def human_te_all(args):
    targets_path = 'human_TE.csv'
    symbol_to_fold_path = 'human_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = ['LL', 'P', 'P5', 'PC', 'P3', 'WP', 'K', 'CF', 'AAF', 'DC', 
                   '1mer_freq_5', '1mer_freq_C', '1mer_freq_3', '2mer_freq_5', '2mer_freq_C', '2mer_freq_3', 
                   '3mer_freq_5', '3mer_freq_C', '3mer_freq_3', '4mer_freq_5', '4mer_freq_C', '4mer_freq_3', 
                   '5mer_freq_5', '5mer_freq_C', '5mer_freq_3', '6mer_freq_5', '6mer_freq_C', '6mer_freq_3', 
                   'Struct']

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/human/TE/',
        bio_source='mean_te',
        species='human',
        )   

def human_rna(args):
    targets_path = 'human_RNA.csv'
    symbol_to_fold_path = 'human_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    # feature_set = [
    #     'LL', 'P5', 'P3', 'CF', 'AAF',
    #     '3mer_freq_5', 'Struct'
    # ]
    # feature_set = ['LL', 'P', 'P5', 'PC', 'P3', 'WP', 'K', 'CF', 'AAF', 'DC', 
    #                '1mer_freq_5', '1mer_freq_C', '1mer_freq_3', '2mer_freq_5', '2mer_freq_C', '2mer_freq_3', 
    #                '3mer_freq_5', '3mer_freq_C', '3mer_freq_3', '4mer_freq_5', '4mer_freq_C', '4mer_freq_3', 
    #                '5mer_freq_5', '5mer_freq_C', '5mer_freq_3', '6mer_freq_5', '6mer_freq_C', '6mer_freq_3', 
    #                'Struct']
    feature_set = [
        'LL', 'P5', 'P3', 'CF', 'AAF',
        '3mer_freq_5'
    ]

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/human/RNA/',
        bio_source='mean_te',
        species='human',
        )   

def human_rna_all(args):
    targets_path = 'human_RNA.csv'
    symbol_to_fold_path = 'human_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = ['LL', 'P', 'P5', 'PC', 'P3', 'WP', 'K', 'CF', 'AAF', 'DC', 
                   '1mer_freq_5', '1mer_freq_C', '1mer_freq_3', '2mer_freq_5', '2mer_freq_C', '2mer_freq_3', 
                   '3mer_freq_5', '3mer_freq_C', '3mer_freq_3', '4mer_freq_5', '4mer_freq_C', '4mer_freq_3', 
                   '5mer_freq_5', '5mer_freq_C', '5mer_freq_3', '6mer_freq_5', '6mer_freq_C', '6mer_freq_3', 
                   'Struct']

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/human/RNA/',
        bio_source='mean_te',
        species='human',
        )   

def mouse_corr(args):
    targets_path = 'mouse_RNA_TE_corr.csv'
    symbol_to_fold_path = 'mouse_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = [
        'LL', 'P5', 'P3', 'CF', 'AAF',
        '3mer_freq_5'
    ]

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/mouse/RNA_TE_corr/',
        bio_source='bio_source_TE_RNA_cor_value_nond',
        species='mouse',
        )    

def mouse_corr_all(args):
    targets_path = 'mouse_RNA_TE_corr.csv'
    symbol_to_fold_path = 'mouse_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = ['LL', 'P', 'P5', 'PC', 'P3', 'WP', 'K', 'CF', 'AAF', 'DC', 
                   '1mer_freq_5', '1mer_freq_C', '1mer_freq_3', '2mer_freq_5', '2mer_freq_C', '2mer_freq_3', 
                   '3mer_freq_5', '3mer_freq_C', '3mer_freq_3', '4mer_freq_5', '4mer_freq_C', '4mer_freq_3', 
                   '5mer_freq_5', '5mer_freq_C', '5mer_freq_3', '6mer_freq_5', '6mer_freq_C', '6mer_freq_3', 
                   'Struct']

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/mouse/RNA_TE_corr/',
        bio_source='bio_source_TE_RNA_cor_value_nond',
        species='mouse',
        )    
    
def mouse_te(args):
    targets_path = 'mouse_TE.csv'
    symbol_to_fold_path = 'mouse_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = [
        'LL', 'P5', 'P3', 'CF', 'AAF',
        '3mer_freq_5'
    ]

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/mouse/TE/',
        bio_source='mean_te',
        species='mouse',
        )
    
def mouse_te_all(args):
    targets_path = 'mouse_TE.csv'
    symbol_to_fold_path = 'mouse_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = ['LL', 'P', 'P5', 'PC', 'P3', 'WP', 'K', 'CF', 'AAF', 'DC', 
                   '1mer_freq_5', '1mer_freq_C', '1mer_freq_3', '2mer_freq_5', '2mer_freq_C', '2mer_freq_3', 
                   '3mer_freq_5', '3mer_freq_C', '3mer_freq_3', '4mer_freq_5', '4mer_freq_C', '4mer_freq_3', 
                   '5mer_freq_5', '5mer_freq_C', '5mer_freq_3', '6mer_freq_5', '6mer_freq_C', '6mer_freq_3', 
                   'Struct']

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/mouse/TE/',
        bio_source='mean_te',
        species='mouse',
        )

def mouse_rna(args):
    targets_path = 'mouse_RNA.csv'
    symbol_to_fold_path = 'mouse_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = [
        'LL', 'P5', 'P3', 'CF', 'AAF',
        '3mer_freq_5'
    ]

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/mouse/RNA/',
        bio_source='mean_te',
        species='mouse',
        )
    
def mouse_rna_all(args):
    targets_path = 'mouse_RNA.csv'
    symbol_to_fold_path = 'mouse_symbol_to_fold.tsv'

    create_symbol_to_fold("./data/" + targets_path, "./data/" + symbol_to_fold_path)

    model_name = 'lgbm'

    feature_set = ['LL', 'P', 'P5', 'PC', 'P3', 'WP', 'K', 'CF', 'AAF', 'DC', 
                   '1mer_freq_5', '1mer_freq_C', '1mer_freq_3', '2mer_freq_5', '2mer_freq_C', '2mer_freq_3', 
                   '3mer_freq_5', '3mer_freq_C', '3mer_freq_3', '4mer_freq_5', '4mer_freq_C', '4mer_freq_3', 
                   '5mer_freq_5', '5mer_freq_C', '5mer_freq_3', '6mer_freq_5', '6mer_freq_C', '6mer_freq_3', 
                   'Struct']

    metrics = [r2_score, pearsonr, spearmanr, mean_squared_error]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/mouse/RNA/',
        bio_source='mean_te',
        species='mouse',
        )
    

# classification
def human_corr_class(args):
    targets_path = 'human_RNA_TE_corr.csv'
    symbol_to_fold_path = 'human_class_balance_symbol_to_fold.tsv'

    add_symbol_to_fold_with_class_balance(f'./data/data_with_{targets_path}', "./data/" + symbol_to_fold_path)

    model_name = 'lgbm_classification'

    feature_set = [
        'LL', 'P5', 'P3', 'CF', 'AAF',
        '3mer_freq_5'
    ]
    # feature_set = [
    #     'LL', 'P5', 'P3', 'CF', 'AAF',
    #     '3mer_freq_5', 'Struct', 'Biochem'
    # ]

    metrics = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/human/RNA_TE_corr_classification/',
        bio_source='bio_source_buffering_category',
        species='human',
        )    

def human_corr_class_all(args):
    targets_path = 'human_RNA_TE_corr.csv'
    symbol_to_fold_path = 'human_class_balance_symbol_to_fold.tsv'

    add_symbol_to_fold_with_class_balance(f'./data/data_with_{targets_path}', "./data/" + symbol_to_fold_path)

    model_name = 'lgbm_classification'

    feature_set = ['LL', 'P', 'P5', 'PC', 'P3', 'WP', 'K', 'CF', 'AAF', 'DC', 
                '1mer_freq_5', '1mer_freq_C', '1mer_freq_3', '2mer_freq_5', '2mer_freq_C', '2mer_freq_3', 
                '3mer_freq_5', '3mer_freq_C', '3mer_freq_3', '4mer_freq_5', '4mer_freq_C', '4mer_freq_3', 
                '5mer_freq_5', '5mer_freq_C', '5mer_freq_3', '6mer_freq_5', '6mer_freq_C', '6mer_freq_3', 
                'Struct']

    metrics = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score]

    name = get_model_dir_name(model_name, feature_set)
    run_experiment(
        args=args,
        TE_path=targets_path, 
        symbol_to_fold_path=symbol_to_fold_path, 
        model_name=model_name, 
        features_to_extract=feature_set,
        metrics=metrics,
        name=name,
        results_path='results/human/RNA_TE_corr_classification/',
        bio_source='bio_source_buffering_category',
        species='human',
        )    

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    tqdm.pandas()
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, required=True)
    parser.add_argument('--permutation_fi', '-p', action='store_true', help='Whether to run permutation feature importance')
    parser.add_argument('--save', '-s', action='store_true', help='Whether to save models')
    args = parser.parse_args()

    if args.experiment in locals():
        locals()[args.experiment](args)
    else:
        raise ValueError(f'Experiment {args.experiment} not found')
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from train import RANDOM_SEED

def add_symbol_to_fold_with_class_balance(data_path, symbol_to_fold_path):
    if not os.path.exists(symbol_to_fold_path):
        data = pd.read_csv(data_path, sep='\t', index_col=None)
        symbol_to_fold = pd.DataFrame(index=data['SYMBOL'], columns=['fold_class_balance'])
        symbol_to_fold.index.name = 'SYMBOL'
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        for fold, (_, test_index) in enumerate(kf.split(data, data['bio_source_buffering_category'])):
            symbol_to_fold.iloc[test_index, symbol_to_fold.columns.get_loc('fold_class_balance')] = fold
        print(symbol_to_fold['fold_class_balance'].value_counts())
        # for each fold print how many are in each class
        if 'fold_class_balance' in data.columns:
            print('Fold class balance already exists in data, overwriting...')
        data = data.merge(symbol_to_fold, left_on='SYMBOL', right_index=True, how='left')
        print(data['fold_class_balance'].value_counts())
        # for each fold print how many are in each class
        print(data.groupby('fold_class_balance')['bio_source_buffering_category'].value_counts())
        symbol_to_fold.to_csv(symbol_to_fold_path, sep='\t')
        data.to_csv(data_path, sep='\t', index=False)
    else:
        print('Symbol to fold with class balance already exists')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, sep='\t', index_col=None)
            if 'fold_class_balance' not in data.columns:
                print('Adding fold class balance to data')
                symbol_to_fold = pd.read_csv(symbol_to_fold_path, sep='\t', index_col=0)
                data['fold_class_balance'] = symbol_to_fold['fold_class_balance']
                data.to_csv(data_path, sep='\t', index=False)
            else:
                print('Fold class balance already exists in data')
        else:
            print('Data does not exist, skipping...')



        
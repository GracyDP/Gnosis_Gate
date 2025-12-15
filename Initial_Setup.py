import gc
import os
import re
import shutil
import sys
import warnings
from itertools import product

import pandas as pd
import tensorflow as tf
import torch
from tqdm import tqdm

from utils.Homomorphic.homomorphic_client import create_context
from utils.Threshold.threshold_client import load_or_create_thresholds, select_items_from_list
from utils.url.uri import load_uri
from utils.utils_Client import get_dataset_pandas, fill_dataset_with_encrypt_to_file_h5, find_all_csv_files
from utils.utils_common import print_status

warnings.filterwarnings("ignore")

evaluate_endpoint = "/evaluate-column"
endpoint_load_lazy = "/load_lazy"
SERVER_URL = load_uri('server')

SCHEMA = 'CKKS'
# SCHEMA = 'BFV'

path_dict = {}

def create_dir_AllExperiment(exp):
    modality = exp['modality']
    dataset_name = exp['name_db']

    numeric_values = [0, 2, 4, 8]
    string_values = [0.50, 0.75, 0.90, 1.0]

    db_n = (dataset_name.replace('.csv', '')).replace('_preProcessed', '')
    path_base = f"./Test/{db_n}/"
    os.makedirs(path_base, exist_ok=True)
    listPath = []

    for num, string in product(numeric_values, string_values):
        path = f'{path_base}' + str(modality) + '/output_N' + str(num) + '_S' + str(string)
        path_test = f'{path}/tmpFiles/'

        os.makedirs(path, exist_ok=True)
        os.makedirs(path_test, exist_ok=True)
        listPath.append(path)

    global path_dict

    path_dict['saveDF'] = listPath
    path_dict['emb_df_h5'] = f'{path_base}{db_n}_Embedded.h5'
    path_dict['df_noEnc'] = f'{path_base}{db_n}_NoEnc.h5'
    path_dict['path_encrypted_h5'] = f'{path_base}' + str(modality) + f'/{db_n}_Encrypted.h5'
    path_dict['path_tmp'] = {
        'original' : path_test,
        'path_tmp_cosine': f'{path_test}Cosine_',
        'path_tmp_diff': f'{path_test}Diff_',
    }

    return db_n, path_dict


sensitive_cols_manuale = None

def main(experiments):

    for exp in tqdm(experiments, total= len(experiments), desc=f'Elaborazione Exp', leave=False, colour='cyan', disable=True):

        sys.stdout.flush()
        modality = exp['modality']
        dataset_name = exp['name_db']
        dataset_path = exp['dataset_path']
        calculate_thr = exp['calculate_thr']

        db_n, path_dict = create_dir_AllExperiment(exp)
        print(f"[CLIENT] Select: {dataset_name} - Len {(pd.read_csv(dataset_path)).shape} - Modality: {modality}\n")

        for v in path_dict['saveDF']:
            if modality != 'All_NOT_Sensitive':
                if SCHEMA == 'CKKS':
                    print_status('[CLIENT]','[CKKS] Creating CKKS context...')
                    create_context(SCHEMA, v)
                elif SCHEMA == 'BFV':
                    print_status('[CLIENT]','[BFV] Creating BFV context...')
                    create_context(SCHEMA, v)

        df = get_dataset_pandas(dataset_path)

        for v in path_dict['saveDF']:
            m = re.search(r'output_N(\d+)_S(.+)', v)

            if m:
                num_val = int(m.group(1))  # numero
                string_val = m.group(2)  # stringa
            else:
                raise Exception('Valori non trovati')

            threshold = {
                    'numeric': num_val,
                    'string': string_val
                }

            global sensitive_cols_manuale

            if sensitive_cols_manuale is None:
                columns_with_types = {
                    col: (
                        # 'boolean' if df[col].dtype == bool else
                        'numeric' if pd.api.types.is_numeric_dtype(df[col]) else
                        'string'
                    )
                    for col in tqdm(df.columns,
                                    desc='[CLIENT][Thresholds]Loading Columns types',
                                    total=len(df.columns),
                                    leave=False,
                                    disable=True)
                }
                sensitive_cols_manuale = select_items_from_list(columns_with_types)

            load_or_create_thresholds(df, v, calculate_thr, threshold,dataset_name, modality, sensitive_cols_manuale)

        if fill_dataset_with_encrypt_to_file_h5(input_df=df, path_dict=path_dict):
            print_status('[CLIENT]',' Homomorphic Transformation Done.')
        else:
            raise Exception('[CLIENT] Error while transforming dataset.')

        # 2. Cancella variabili pesanti
        del df

        # 3. Garbage collector
        gc.collect()

        # 4. Reset TensorFlow / PyTorch se serve
        tf.keras.backend.clear_session()  # TF
        torch.cuda.empty_cache()  # PyTorch

        sys.stdout.flush()

    print('\n All Dataset has been created...\n')

if __name__ == '__main__':

    list_dataset = find_all_csv_files("DB")

    modality = ['Manual','All_NOT_Sensitive', 'All_Sensitive']

    # Generazione delle configurazioni
    experiments = []

    for dataset_path in list_dataset:
        db_name = dataset_path.split('\\')[-1]
        for mod in modality:
            config = {
                'name_db': db_name,
                'dataset_path': dataset_path,
                'modality': mod,
                'threshold': {
                    'numeric': 0,
                    'string': 'None'
                },
                'calculate_thr': True,
                'create_context': True,
                'create_dataset': True,
            }
            experiments.append(config)
    print(f'[START] Collected {len(experiments)//len(list_dataset)} experiments for {len(list_dataset)} datasets!\n')

    if os.path.exists('Test'):
        shutil.rmtree('Test', ignore_errors=True)

    main(experiments)

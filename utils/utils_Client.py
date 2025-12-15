import glob
import multiprocessing
import tracemalloc
import warnings
warnings.filterwarnings("ignore")

import h5py
import json
import torch
from memory_profiler import memory_usage
from sentence_transformers import SentenceTransformer

from utils.Homomorphic.homomorphic_client import invia_bfv, invia_ckks, decode_decrypt_ckks, ricevi_ckks, \
    convert_from_object_to_ckks, convert_from_object_to_bfv
from utils.Threshold.threshold_client import get_thr, get_AllStr_columns, get_AllSens_columns, \
    get_Numeric_NoSens, get_String_NoSens, get_NoSensCol_from_list

from utils.processing_column.similarity_partition import similarity_numeric_gpu, similarity_string_gpu
from utils.utils_common import load_h5_file, load_columns_from_h5, save_df_h5, \
    print_status

import ast
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import gc

batch_size = 5000
batch_update = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device, token='HuggingFace_TOKEN')

def normalize(vec):

    # Converte la stringa in una lista Python e Converte la lista in un array NumPy
    if isinstance(vec, str):
        vec = ast.literal_eval(vec)
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def scale_and_convert_to_int(scale_emb, scale=1000):
    return np.round(np.array(ast.literal_eval(scale_emb)) * scale).astype(np.int64)

def fill_dataset_with_embedd(df, df_path):
    string_cols = get_AllStr_columns()
    pbar = tqdm(total=len(string_cols), desc='Embedding columns', leave=False, colour='magenta')

    for col, info in string_cols:
        df[col].fillna('', inplace=True)
        total_len = len(df)

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(embed_series, df[col].iloc[start:end].tolist()): (start, end)
                for start in range(0, total_len, batch_size)
                for end in [min(start + batch_size, total_len)]
            }

            for f in as_completed(futures):
                start, end = futures[f]
                embeddings = f.result()
                df[col].iloc[start:end] = embeddings.tolist()  # scrittura diretta
        pbar.update(1)

    if df_path.endswith('.h5'):
        save_df_h5(df_path=df_path, df=df)

    return df

def encrypt_batch(batch_values, type_data, schema):
    result = []

    if type_data == 'numeric':
        if schema == 'BFV':
            result = [invia_bfv(convert_from_object_to_bfv([v]))
                for v in batch_values
            ]
        elif schema == 'CKKS':
            result = [invia_ckks(convert_from_object_to_ckks([v]))
                for v in batch_values
            ]
    elif type_data == 'string':
        if schema == 'BFV':
            int_vectors = [scale_and_convert_to_int(val) for val in batch_values]
            result = [invia_bfv(convert_from_object_to_bfv(vec))
                for vec in int_vectors
            ]
        elif schema == 'CKKS':
            norm_vectors = [normalize(vec) for vec in batch_values]
            result = [invia_ckks(convert_from_object_to_ckks(vec))
                for vec in norm_vectors
            ]
    else:
        raise Exception("Unsupported type_data")

    return result

def fill_dataset_with_encrypt_to_file_h5(input_df, path_dict,
                                      schema='CKKS'):
    """
    Cripta tutte le colonne sensibili e salva i risultati in file separati.
    - Le colonne normali vengono salvate una sola volta.
    - Ogni colonna sensibile cifrata viene salvata in un file separato.
    - Alla fine puoi ricombinare tutto in un unico Parquet.
    """

    emb_df_path = path_dict['emb_df_h5']
    path_dfH5  = path_dict['path_encrypted_h5']

    # Carica o genera embeddings
    if not os.path.exists(emb_df_path):
        df = fill_dataset_with_embedd(input_df, emb_df_path)
    else:
        if emb_df_path.endswith('.h5'):
            df = load_h5_file(emb_df_path)

    # Identifica colonne sensibili e normali
    sensitive_cols = get_AllSens_columns()

    if not sensitive_cols:
        print_status('[CLIENT]',"Nessuna colonna sensibile trovata. File copiato senza cifratura.")
        save_df_h5(df_path=path_dict['df_noEnc'], df=df)
        del df
        return True
    else:
        print_status('[CLIENT]',f'Trovate {len(sensitive_cols)} Colonne Sensibili')

    normal_cols = get_NoSensCol_from_list(df.columns)

    # Apri HDF5 in modalità append
    store = pd.HDFStore(path_dfH5, key='df',mode='w')

    # Salva colonne normali se presenti
    if normal_cols:
        for n_col in normal_cols:
            store[n_col] = df[n_col]
            df.drop(columns=[n_col], inplace=True)
        print_status('[CLIENT]',f"Salvate {len(normal_cols)} Colonne non CKKS")
    else:
        print_status('[CLIENT]',"Tutte le colonne sono da cifrare")

    df_len = len(df)
    gc.collect()

    min_chunk_threshold = 1000

    if df_len <= min_chunk_threshold:
        num_workers = 2
        batch_size = df_len // 2  # batch uguale a tutto, flush solo finale
    else:
        num_workers = min(multiprocessing.cpu_count(), df_len)
        batch_size = max(df_len // num_workers, 500)

    # Cripta e salva colonne sensibili
    if sensitive_cols:
        # Cripta e salva colonne sensibili
        with ProcessPoolExecutor() as executor:
            pbar = tqdm(total=len(sensitive_cols), desc="Encrypting DB", colour="cyan", leave=False)

            for col, info in sensitive_cols:
                type_data = info.get('type')
                col_hdf_key = col

                if col_hdf_key not in store.keys():
                    futures = {}
                    for start in range(0, df_len, batch_size):
                        end = min(start + batch_size, df_len)
                        f = executor.submit(encrypt_batch, df[col].values[start:end], type_data, schema)
                        futures[f] = (start, end)

                    encrypted_col = []

                    for f in as_completed(futures):
                        start, end = futures[f]
                        encrypted_values = f.result()
                        encrypted_col.extend(encrypted_values)
                        del encrypted_values
                        gc.collect()

                    # Salva la colonna completa in HDF5
                    store.put(col_hdf_key, pd.DataFrame({col: encrypted_col}), format='table',complib='blosc',complevel=5)
                    del encrypted_col
                    gc.collect()

                pbar.update(1)
    else:
        print_status('[CLIENT]',"Nessuna colonna da criptare")

    store.close()
    gc.collect()
    print_status('[CLIENT]',"Encrypting done -> saved in HDF5")

    return True

def embed_series(texts):
    """
    Convert a pandas Series of text to a Series of embeddings.
    """
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size
    )
    return embs

def compute_similarity_no_sens(original_col, path_df):
    """
    Compute non-sensitive similarity on encrypted DataFrame,
    freeing memory promptly after each step.
    """
    result_numeric = {}
    result_string = {}

    print_status('[CLIENT]','[CLIENT][NO Sens Similarity] Compute similarity scores..')
    numeric_no_sens = get_Numeric_NoSens(original_col)

    string_no_sens = get_String_NoSens(original_col)

    if len(numeric_no_sens) > 0:
        print_status('[CLIENT]',f'[Numeric No Sens] Found {len(numeric_no_sens)} Columns...')

        # 1) Similarity numeric
        result_numeric = similarity_numeric_gpu(load_columns_from_h5(path_df, numeric_no_sens), original_col)
        # libera i dati numerici non più necessari

        if result_numeric is not None:
            print_status('[CLIENT]','[Numeric] Similarity for numeric Done...')
        else:
            raise Exception('[Numeric] Error Similarity for numeric')

    elif len(numeric_no_sens) == 0 or not numeric_no_sens:
        print_status('[CLIENT]',f'[Numeric No Sens] No Columns Found...')

    if len(string_no_sens) > 0:
        print_status('[CLIENT]',f'[String No Sens] Found {len(string_no_sens)} Columns...')

        # 1) Similarity string
        result_string = similarity_string_gpu(load_columns_from_h5(path_df, string_no_sens), original_col)
        # libera i dati string non più necessari

        if result_string is not None:
            print_status('[CLIENT]','[String] Similarity for string Done...')
        else:
            raise Exception('[String] Error Similarity for string')

    elif len(string_no_sens) == 0 or not string_no_sens:
        print_status('[CLIENT]',f'[String No Sens] No Columns Found...')

    print_status('[CLIENT]','[NO Sens Similarity] Done..')

    gc.collect()
    return result_numeric | result_string

def batcher(iterable, tipo):
    """Generator per creare batch da un iterable."""
    batch = []
    for item in iterable:
        diff_enc = item[tipo]
        try:
            item[tipo] = decode_decrypt_ckks(ricevi_ckks(diff_enc))
        except Exception:
            continue

        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

base_pth_dataset = ''

def get_dataset_pandas(dataset_name, file_type = 'csv'):

    path = f'{base_pth_dataset}{dataset_name}'

    return ((pd.read_csv(path))
          if file_type == 'csv'
          else pd.read_parquet(path))

def load_parquet_file(path):
    df = pd.read_parquet(path)
    return df

def find_all_csv_files(root_dir):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                full_path = os.path.join(dirpath, filename)
                csv_files.append(full_path)

    if os.path.exists('Test'):
        for dirpath, _, filenames in os.walk('Test'):
            for filename in filenames:
                if 'Exp_Errors.txt' == filename:
                    full_path = os.path.join(dirpath, filename)
                    os.remove(full_path)

    return csv_files

def read_h5_to_df(h5_path, tipo=None):
    """
    Legge un file HDF5 generato dal writer_thread e ritorna DataFrame.

    Args:
        h5_path (str): percorso del file HDF5.
        tipo (str, opzionale): nome del dataset da leggere.
                               Se None, legge tutti i dataset.

    Returns:
        dict[str, pd.DataFrame]: dizionario {tipo: DataFrame}
    """
    if not os.path.exists(h5_path):
        raise  Exception(f"File non trovato: {h5_path}")

    dfs = {}
    with h5py.File(h5_path, 'r') as h5_file:
        tipi = [tipo] if tipo else list(h5_file.keys())

        for t in tipi:
            if t not in h5_file:
                raise Exception(f"Dataset '{t}' non trovato.")

            dataset = h5_file[t][:]
            records = [json.loads(x) for x in dataset if x]

            if not records:
                dfs[t] = pd.DataFrame()
                continue

            dfs[t] = pd.DataFrame(records)

    return (dfs[t]).to_dict(orient="records")

def create_dir(exp):
    modality = exp['modality']
    dataset_name = exp['name_db']
    value_thr = exp['threshold']

    db_n = (dataset_name.replace('.csv', '')).replace('_preProcessed', '')
    path_base = f"./Test/{db_n}/"
    # path_base_test = f"./Test/{db_n}/tmp/"
    os.makedirs(path_base, exist_ok=True)
    # os.makedirs(path_base_test, exist_ok=True)

    path = f'{path_base}' + str(modality) + '/output_N' + str(value_thr['numeric']) + '_S' + str(
        value_thr['string'])

    path_test = f'{path}/tmpFiles/'

    os.makedirs(path, exist_ok=True)
    os.makedirs(path_test, exist_ok=True)
    path_dict = {}
    path_dict['emb_df_h5'] = f'{path_base}{db_n}_Embedded.h5'
    path_dict['df_noEnc'] = f'{path_base}{db_n}_NoEnc.h5'
    path_dict['path_encrypted_h5'] = f'{path_base}' + str(modality) + f'/{db_n}_Encrypted.h5'
    path_dict['path_tmp'] = {
        'original' : path_test,
        'path_tmp_cosine': f'{path_test}Cosine_',
        'path_tmp_diff': f'{path_test}Diff_',
    }

    if modality == 'All_NOT_Sensitive':
        path_dict['path_df'] = path_dict['df_noEnc']
    else:
        path_dict['path_df'] = path_dict['path_encrypted_h5']

    gc.collect()
    return db_n, path, path_dict
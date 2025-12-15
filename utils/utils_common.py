import ast
import gc
import sys
import time

import numpy as np
import pandas as pd
import tenseal as ts
import tensorflow as tf
import torch

from utils.Threshold.threshold_client import get_String_NoSens, get_AllStr_columns

def identifica_tipo(elem):
    numeric_types = (int, float, np.integer, np.floating)

    if isinstance(elem, ts.CKKSVector):
        return "CKKSVector"

    elif isinstance(elem, str):
        try:
            parsed = ast.literal_eval(elem)
            if (isinstance(parsed, list)) and all(isinstance(x, numeric_types) for x in parsed):
                return "Embeddings"
        except (ValueError, SyntaxError) as e:
            pass
        return "String"

    elif isinstance(elem, numeric_types):
        return "Numeric"

    elif (isinstance(elem, list) or isinstance(elem, np.ndarray)) and all(isinstance(x, numeric_types) for x in elem):
        return "ndarray"

    else:
        return "Tipo sconosciuto"

def get_column_names_from_h5(path):
    try:
        # Usa solo i metadati per ottenere i nomi delle colonne
        with pd.HDFStore(path, mode='r') as store:
            if 'df' in store:
                # Recupera solo le colonne dai metadati (non carica i dati)
                return store.get_storer('df').attrs.non_index_axes[0][1]
            else:
                return [key.strip('/') for key in store.keys()]
    except Exception as e:
        raise Exception(f"Errore: {e}")

def load_columns_from_h5(path, columns):
    """
    Carica un sotto-DataFrame da un file HDF5 (.h5), estraendo solo le colonne richieste.

    Args:
        path (str): path al file .h5
        columns (list[str]): lista delle colonne da estrarre
        key (str): nome del dataset salvato (default "df")

    Returns:
        pd.DataFrame: dataframe con solo le colonne richieste
    """

    # Apri HDF5
    store = pd.HDFStore(path, mode='r')
    key_c = ''
    if '/df' in store.keys():
        key_c = ['/df']
    else:
        key_c = store.keys()

    # Leggi tutte le chiavi (dataset) presenti
    all_keys = [key.strip('/') for key in store.keys() if key in key_c]  # rimuove eventuale '/'
    all_keys = [key for key in all_keys if key.replace('col_', '') in columns]  # rimuove eventuale '/'

    # Inizializza DataFrame finale
    df_final = pd.DataFrame()

    # Carica tutti i dataset nell’ordine in cui sono salvati in HDF5
    for key in all_keys:
        df_final = pd.concat([df_final, store[key]], axis=1)

    store.close()

    string_no_sens = get_String_NoSens(columns)

    if string_no_sens:
        # Applica la sostituzione a tutte le colonne in string_no_sens
        for col in string_no_sens:
            df_final[col] = df_final[col].apply(parse_vector).dropna()
            if df_final[col].empty:
                raise ValueError(f"Nessun embedding valido trovato in colonna {col}")

    return df_final[columns]

def save_df_h5(df_path, df, column_to_save=None):
    with pd.HDFStore(df_path, mode='w') as store:
        columns = column_to_save if column_to_save else df.columns

        for col_key in columns:
            value = df[col_key]

            # Mappa i tipi degli elementi
            types = value.map(type)

            # Trova i tipi unici presenti nella colonna
            unique_types = types.unique()

            if len(unique_types) == 1:
                val = value[types == unique_types[0]].iloc[0]
                if isinstance(val, list) and identifica_tipo(val) == 'ndarray':
                    df[col_key] = df[col_key].apply(lambda x: str(x))
                elif identifica_tipo(val) == 'Numeric' or identifica_tipo(val) == 'Embeddings':
                    pass
                else:
                    raise Exception('Valori non trovati')
                store.put(f'col_{col_key}', df[col_key], format='table', complib='blosc', complevel=5)
            elif len(unique_types) > 1 or not unique_types:
                raise Exception(f'Column {col_key} contains multiple types: {types}')

    return True

def load_h5_file(hdf5_path):
    # Apri HDF5
    store = pd.HDFStore(hdf5_path, mode='r')

    # Leggi tutte le chiavi (dataset) presenti
    all_keys = [key.strip('/') for key in store.keys()]  # rimuove eventuale /
    # Inizializza DataFrame finale
    df_final = pd.DataFrame()

    # Carica tutti i dataset nell’ordine in cui sono salvati in HDF5
    for key in all_keys:
            df_final = pd.concat([df_final, store[key]], axis=1)

    store.close()

    string = [col_df for col_df in df_final.columns if
                      col_df in [c for c, _ in get_AllStr_columns()]]

    for colS in string:
        df_final[colS] = df_final[colS].apply(lambda x: ast.literal_eval(x) if identifica_tipo(x)=="Embeddings" else x)

    if df_final.empty:
        raise Exception('No data found')

    return df_final

def parse_vector(x, decimals=None):
    if isinstance(x, str):
        x = x.replace(",,", ",")
        try:
            arr = np.array(ast.literal_eval(x))
        except Exception:
            return None
    elif isinstance(x, (list, np.ndarray)):
        arr = np.array(x)
    else:
        return None

    if decimals is not None:
        return "[" + ", ".join(f"{v:.{decimals}f}" for v in arr.tolist()) + "]"
    else:
        return str(arr.tolist())

def print_status(app, message):
    color = "\033[36m" if "[CLIENT]" in app else "\033[33m"
    sys.stdout.write(f"{color}{app}{message}\033[0m\n")
    sys.stdout.flush()

def the_end(url):
    # 3. Garbage collector
    gc.collect()

    # 4. Reset TensorFlow / PyTorch se serve
    tf.keras.backend.clear_session()  # TF
    torch.cuda.empty_cache()  # PyTorch

    sys.stdout.flush()
    print(f'\n{url} All Tasks Complete!', end="")

    text = "TO BE CONTINUED"
    print("\n" + "=" * 30)
    print(f"     {text}", end="")
    sys.stdout.flush()

    for i in range(3):
        time.sleep(1)
        print(".", end="")
        sys.stdout.flush()

    print("\n" + "=" * 30)

    text2 = "MADE WITH VENOM"
    text3 = "BY OROCHIMARU"
    print("=" * 30)
    print(f"     {text2}")
    print(f"       {text3}", end="")
    sys.stdout.flush()

    print("\n" + "=" * 30, end="")
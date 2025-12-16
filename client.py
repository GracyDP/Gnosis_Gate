"""
====================================================
Python Client per l’elaborazione sicura di dataset
====================================================
Questo script funge da client che comunica con un server Flask (server.py).
Principali funzionalità:
- Caricare dataset (CSV, ecc.) e inviarli al server
- Creare e inviare un contesto CKKS per computazioni omomorfiche
- Calcolare partizioni (RFDs) e matrici di DCs su dataset
- Gestire thresholds (soglie numeriche e stringhe)
- Monitorare utilizzo memoria e tempi di esecuzione
- Salvare risultati, log e sperimentazioni su file

Pipeline tipica:
1) Caricamento dataset
2) Creazione thresholds
3) Creazione dataset omomorfico (opzionale)
4) Invio dataset e thresholds al server
5) Esecuzione partizioni + minimi
6) Raccolta e salvataggio dei risultati
"""

import gc
import json
import os
import sys
import warnings

import requests

# Funzioni di utilità per crittografia, query e thresholds
from utils.Homomorphic.homomorphic_client import create_context, send_ckks_context
from utils.Query import query_toServer
from utils.Query.query_toServer import delete_file
from utils.Threshold.threshold_client import load_or_create_thresholds, send_threshold, get_SensCol_from_list
from utils.url.uri import load_uri
from utils.utils_Client import get_dataset_pandas, compute_similarity_no_sens, \
    fill_dataset_with_encrypt_to_file_h5, create_dir, \
    find_all_csv_files
from utils.utils_common import print_status

warnings.filterwarnings("ignore")

# Librerie per thread, tempo e combinazioni
from threading import Thread
import time
from itertools import product
import pandas as pd

# Endpoint del server
evaluate_endpoint = "/evaluate-column"
server_url = load_uri('server')

# Schema crittografico da utilizzare
SCHEMA = 'CKKS'
# SCHEMA = 'BFV'

# Dizionario con i path del dataset e file temporanei
path_dict = {}

#**********NEW
def _matrix_already_computed(exp_path: str) -> bool:
    """
    True se la matrice (output server) è già presente nella cartella esperimento.
    """
    matrices_dir = os.path.join(exp_path, "Matrices")
    matrix_csv = os.path.join(exp_path, "Matrix_DCs.csv")
    return os.path.isdir(matrices_dir) and os.path.exists(matrix_csv)
#**********END


def compute_minimum(path):
    print_status('[CLIENT]', ' Inizio computazioni minimi.')

    url = f"{server_url.rstrip('/')}{evaluate_endpoint}"
    #**********NEW
    print_status('[CLIENT]', f' Chiamo server: {url}')
    print_status('[CLIENT]', f' path_df={path_dict.get("path_df")} | path_exp={path}')
    #**********END

    resp = requests.get(url, json={'path_df': path_dict['path_df'], 'path': path})

    #**********NEW
    # Se il server risponde con errore, stampo un minimo di contesto prima di lanciare l’eccezione
    if resp.status_code >= 400:
        print_status('[CLIENT]', f' Server response status={resp.status_code}')
        try:
            print_status('[CLIENT]', f' Server response body={resp.text[:500]}')
        except Exception:
            pass
    #**********END

    resp.raise_for_status()

    print_status('[CLIENT]', ' End computazioni minimi.')

    return True


def thr_computeMin(path, df, dataset_name):
    #**********NEW
    # Se la matrice esiste già (es. run precedente crashato dopo averla generata),
    # evito di richiamare il server e considero l’operazione completata.
    if _matrix_already_computed(path):
        print_status('[CLIENT]', '[Matrix] Matrice già presente su disco. Skip chiamata al server.')
        print_status('[CLIENT]', f' Path matrice: {os.path.join(path, "Matrix_DCs.csv")}')
        # Provo comunque a (ri)scrivere il logGenerale_DC.txt per coerenza
        try:
            with open(f'{path}/logGenerale_DC.txt', 'w') as f1:
                f1.write(
                    f'Dataset: {dataset_name}\n'
                    f'NUMERO RIGHE: {df.shape[0]}\n'
                    f'NUMERO COLONNE: {df.shape[1]}\n'
                    f'NOTE: Matrice già presente, non ricalcolata.\n'
                )
        except Exception as e:
            print_status('[CLIENT]', f'[WARN] Impossibile scrivere logGenerale_DC.txt: {e}')
        return True
    #**********END

    start = time.time()

    if compute_minimum(path=path):
        end_time_dc = time.time()
        timing_dc = end_time_dc - start

        #**********NEW
        print_status('[CLIENT]', '[Matrix] Computazione completata. Scrivo logGenerale_DC.txt.')
        #**********END

        try:
            with open(f'{path}/logGenerale_DC.txt', 'w') as f1:
                f1.write(
                    f'Dataset: {dataset_name}\n'
                    f'NUMERO RIGHE: {df.shape[0]}\n'
                    f'NUMERO COLONNE: {df.shape[1]}\n'
                )
                f1.write(
                    f'START TIME: {start:.3f}\n'
                    f'END TIME:{end_time_dc:.3f}\n'
                    f'TIMING: {timing_dc:.3f}'
                )
        except Exception as e:
            print_status('[CLIENT]', f'[WARN] Impossibile scrivere logGenerale_DC.txt: {e}')

    else:
        raise Exception('[CLIENT] Error while creating matrix!')

    return True


def main(experiments):

    for exp in experiments:
        client_thread = Thread(target=query_toServer.main)
        client_thread.start()

        sys.stdout.flush()
        modality = exp['modality']
        dataset_name = exp['name_db']
        dataset_path = exp['dataset_path']
        value_thr = exp['threshold']
        calculate_thr = exp['calculate_thr']
        create_ctx = exp['create_context']
        create_dataset = exp['create_dataset']

        global path_dict
        db_n, path, path_dict = create_dir(exp)

        if dataset_name:
            print(f"*** Select: {db_n} - Len {(pd.read_csv(dataset_path)).shape} - Modality: {modality} - Thr: {value_thr} ***")
        else:
            print_status('[CLIENT]', " No dataset selected")

        print_status('[CLIENT]', ' Starting process...')

        if modality != 'All_NOT_Sensitive':
            print_status('[CLIENT]', f' Creating {SCHEMA} context...')
            if create_context(SCHEMA, path, create_ctx):
                print_status('[CLIENT]', f' {SCHEMA} Done.')
            else:
                print_status('[CLIENT]', f' Failed to create {SCHEMA} context.')

        print_status('[CLIENT]', '[Dataset] Loading original dataset...')
        df = get_dataset_pandas(dataset_path)
        print_status('[CLIENT]', '[Dataset] Done.')

        print_status('[CLIENT]', '[Threshold] Loading thresholds...')
        load_or_create_thresholds(df, path, calculate_thr, value_thr, dataset_name, modality)
        print_status('[CLIENT]', '[Threshold] Done.')

        ### NELLA VERSIONE DI TEST LE THRESHOLD e I DATASET SONO STATI GIA CALCOLATI
        if create_dataset:
            print_status('[CLIENT]', '[Homomorphic] Transforming dataset...')
            if fill_dataset_with_encrypt_to_file_h5(input_df=df, path_dict=path_dict):
                print_status('[CLIENT]', ' Homomorphic Transformation Done.')
            else:
                raise Exception('[CLIENT] Error while transforming dataset.')
        else:
            # Caricare il file parquet in un DataFrame
            print_status('[CLIENT]', '[Mongo] Carico DB da locale.')

        print_status('[CLIENT]', '[Mongo_DB Fake] Upload dataset...')

        print_status('[CLIENT]', '[Threshold] Sending threshold...')
        if send_threshold(path):
            print_status('[CLIENT]', '[Threshold] Threshold Sent.')
        else:
            raise Exception('[CLIENT][Threshold] Failed to send threshold.')

        if modality != 'All_NOT_Sensitive':
            print_status('[CLIENT]', '[CKKS_Context] Sending CKKS_Context...')
            if send_ckks_context():
                print_status('[CLIENT]', '[CKKS_Context] CKKS_Context Sent.')
            else:
                raise Exception('[CLIENT][CKKS_Context] Failed to send CKKS_Context.')

        fx = open("log/log_client.txt", "w")

        if not thr_computeMin(path, df, dataset_name):
            raise Exception('Error while computing Matrix!')

        # --- Shutdown client thread and free resources ---
        client_thread.join(1)
        del client_thread

        fx.close()

        del fx, df
        gc.collect()

        if delete_file(experiments.index(exp) == len(experiments) - 1):
            with open(f'./Test/{db_n}/Exp_Conclusi.txt', 'a') as f:
                f.write(f"Dataset: {db_n} - Modality: {modality} - Thr: {value_thr} - Path: {path}\n")
        else:
            raise Exception('Errore salvataggio esperimenti non conclusi')

        db_n, path, _ = create_dir(exp)

        with open(f'./Test/{db_n}/Exp_Errors.txt', 'a') as f:
            f.write(f"Dataset: {db_n} - Modality: {modality} - Thr: {value_thr} - Path: {path}\n")
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'DONE: Modality: {modality} on {db_n} - Thr: {value_thr}\n')
        print('*' * 30)
        time.sleep(2)


if __name__ == '__main__':
    list_dataset = find_all_csv_files("DB")

    numeric_values = [0, 2, 4, 8]
    string_values = [0.50, 0.75, 0.90, 1.0]
    modality = ['Manual', 'All_NOT_Sensitive', 'All_Sensitive']

    # Generazione delle configurazioni
    experiments = []

    for dataset_path in list_dataset:
        db_name = dataset_path.split('\\')[-1]
        for num, string, mod in product(numeric_values, string_values, modality):
            config = {
                'name_db': db_name,
                'dataset_path': dataset_path,
                'threshold': {
                    'numeric': num,
                    'string': string
                },
                'modality': mod,
                'calculate_thr': False,
                'create_context': False,
                'create_dataset': False,
            }
            experiments.append(config)

    print(f'Collected {len(experiments)//len(list_dataset)} experiments for {len(list_dataset)} datasets!\n')
    main(experiments)

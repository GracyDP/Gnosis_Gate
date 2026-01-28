import ast
import base64
import gc
import json
import multiprocessing
import os
import queue
import threading
import time
import warnings
from concurrent.futures import as_completed, ThreadPoolExecutor
from math import ceil

import faiss
import h5py
import numpy as np
from numba import cuda
from tqdm.auto import tqdm

from utils.Homomorphic.homomorphic_client import ricevi_ckks, convert_from_object_to_ckks
from utils.Threshold.threshold_client import get_thr, get_NumericSens, get_StringSens

from utils.utils_Server import export_columns_to_h5_streaming, load_all_columns_hdf5
from utils.utils_common import load_columns_from_h5, parse_vector, get_column_names_from_h5, print_status

import psutil

warnings.filterwarnings("ignore")

batch_size = 5000
batch_update = 1000
stop_thread = False

colors = ["red", "green", "yellow", "blue", "magenta", "cyan"]
enc_vectors_global = None
total_memory_gb = round(psutil.virtual_memory().total / (1024 ** 3))

@cuda.jit  # Decorator che indica a Python che questa funzione va eseguita sulla GPU
def gpu_compute_similarity(data, thresh_arr, sim_mat, n_rows, n_cols):
    """
    KERNEL GPU: Questa funzione viene eseguita in parallelo da migliaia di thread sulla GPU.
    
    COSA FA:
    Ogni thread confronta DUE RIGHE di UNA COLONNA e decide se sono simili.
    La decisione si basa sulla DIFFERENZA ASSOLUTA tra i valori:
    - Se |valore1 - valore2| <= soglia → simili (1)
    - Altrimenti → non simili (0)
    
    ESEMPIO:
    Colonna "età": [25, 28, 50, 27]
    Soglia: 5 anni
    
    Confronti:
    - riga 0 (25) vs riga 1 (28): |25-28| = 3 <= 5 → simili (1)
    - riga 0 (25) vs riga 2 (50): |25-50| = 25 > 5 → non simili (0)
    - riga 0 (25) vs riga 3 (27): |25-27| = 2 <= 5 → simili (1)
    E così via per tutte le coppie...
    """
    # STEP 1: Calcola gli indici del thread corrente nella griglia 3D
    # Ogni thread ha un ID univoco che determina quale confronto deve fare
    
    # c = indice della colonna da elaborare
    c = cuda.blockIdx.z
    
    # i = indice della prima riga da confrontare
    # Formula: (blocco * dimensione_blocco) + thread_locale
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    # j = indice della seconda riga da confrontare
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    # STEP 2: Controllo dei limiti - assicurati che gli indici siano validi
    # (possono esserci thread "extra" fuori dai limiti dei dati)
    if c < n_cols and i < n_rows and j < n_rows:
        
        # STEP 3: Recupera i valori da confrontare
        # I dati sono memorizzati in un array 1D appiattito,
        # quindi calcoliamo l'indice: riga * num_colonne + colonna
        v_i = data[i * n_cols + c]  # Valore della riga i, colonna c
        v_j = data[j * n_cols + c]  # Valore della riga j, colonna c
        
        # STEP 4: Recupera la soglia per questa colonna
        thresh = thresh_arr[c]
        
        # STEP 5: CALCOLO DELLA SIMILARITY
        # Confronta la differenza assoluta con la soglia:
        # - Se la differenza è <= soglia → i valori sono simili → scrivi 1
        # - Altrimenti → i valori NON sono simili → scrivi 0
        #
        # Esempio con thresh=10:
        #   abs(100 - 105) = 5 <= 10 → simili (1)
        #   abs(100 - 150) = 50 > 10 → non simili (0)
        sim_mat[c, i, j] = 1 if abs(v_i - v_j) <= thresh else 0

def similarity_numeric_gpu(df_num, original_col):
    """
    FUNZIONE PRINCIPALE: Calcola la similarity tra valori numerici usando la GPU.
    
    DIFFERENZA CON COSINE SIMILARITY:
    Questa funzione NON usa la cosine similarity classica, ma una similarity
    basata sulla DISTANZA tra valori: due valori sono simili se la loro
    differenza è minore di una soglia (threshold).
    
    Esempio: se threshold=10 e hai valori 100 e 105, sono simili perché |100-105|=5 < 10
    
    PERCHÉ SU GPU?
    Per dataset grandi con migliaia di righe, calcolare la similarity tra
    ogni coppia di righe richiederebbe molto tempo su CPU.
    La GPU può fare migliaia di confronti in parallelo, velocizzando enormemente.
    """
    # STEP 1: Carica le soglie (threshold) per ogni colonna
    # Ogni colonna ha una soglia che definisce "quanto simili" devono essere due valori
    thresholds = get_thr()

    # STEP 2: Preparazione dei dati
    cols = df_num.columns
    n_rows, n_cols = df_num.shape[0], len(cols)  # Numero di righe e colonne

    # Crea un array 2D (matrice) per contenere tutti i dati numerici
    # Usiamo float32 per risparmiare memoria sulla GPU
    data = np.zeros((n_rows, n_cols), dtype=np.float32)
    thresh_list = []  # Lista delle soglie per ogni colonna
    mapping = {}  # Mappa tra indici e nomi delle colonne

    # STEP 3: Carica i dati colonna per colonna
    pbar = tqdm(desc='[CLIENT] Loading numeric columns', total=n_cols, leave=False)
    for idx, col in enumerate(cols):
        info = thresholds.get(col)  # Informazioni sulla colonna
        mapping[str(original_col.index(col))] = str(col)  # Salva la mappatura

        # Se la colonna è numerica e NON sensibile, carica i suoi valori
        if info and info.get('type') == 'numeric' and not info.get('sensitive'):
            data[:, idx] = (df_num[col].to_numpy().astype(np.float32))

        # Aggiungi la soglia: 0.0 se non c'è info o è sensibile, altrimenti usa la soglia definita
        if not info or info.get('type') != 'numeric' or info.get('sensitive'):
            thresh_list.append(0.0)
        else:
            thresh_list.append(float(info.get('threshold', 0)))
        pbar.update(1)

    # Converti la lista di soglie in array numpy
    thresh_arr = np.array(thresh_list, dtype=np.float32)

    # STEP 4: CONFIGURAZIONE GPU - definisci come organizzare il lavoro parallelo
    # La GPU lavora con "blocchi" e "griglie" di thread (unità di esecuzione parallele)
    dev = cuda.get_current_device()  # Ottieni la GPU corrente
    max_thr = dev.MAX_THREADS_PER_BLOCK  # Massimo numero di thread per blocco
    side = int(np.floor(np.sqrt(max_thr)))  # Calcola dimensione ottimale per blocco quadrato
    
    # Definisci dimensioni blocco: (righe, colonne, profondità)
    bx, by = min(side, n_rows), min(side, n_rows)
    block = (bx, by, 1)
    
    # Definisci dimensioni griglia: quanti blocchi servono per coprire tutti i dati
    # ceil = arrotonda per eccesso per assicurarsi di coprire tutti i dati
    grid = (ceil(n_rows / bx), ceil(n_rows / by), n_cols)

    # STEP 5: TRASFERIMENTO DATI SULLA GPU
    # Copia i dati dalla RAM del computer alla memoria della GPU
    d_data = cuda.to_device(data.ravel())  # Dati appiattiti in 1D
    d_thresh = cuda.to_device(thresh_arr)  # Soglie
    # Alloca spazio sulla GPU per la matrice di similarity (risultato)
    # Dimensione: (n_cols, n_rows, n_rows) - per ogni colonna, matrice n_rows x n_rows
    # Usiamo int8 (1 byte) perché ci basta 0 o 1 (simile/non simile)
    d_sim = cuda.device_array((n_cols, n_rows, n_rows), dtype=np.int8)

    # STEP 6: ESECUZIONE DEL KERNEL GPU
    # Lancia la funzione gpu_compute_similarity sulla GPU
    # La GPU eseguirà migliaia di confronti in parallelo
    # Ogni thread confronterà due righe di una colonna e deciderà se sono simili
    gpu_compute_similarity[grid, block](d_data, d_thresh, d_sim, n_rows, n_cols)

    # Stream results to JSONL per column sequentially
    export_columns_to_h5_streaming(d_sim)

    # Cleanup GPU and host resources
    del d_data, d_thresh, d_sim, data, thresh_list, thresh_arr, df_num,
    cuda.current_context().deallocations.clear()
    gc.collect()

    # Return metadata since data persists on disk and Converti gli indici delle colonne nei nomi
    result = {original_col.index(cols[int(k)]): v for k, v in load_all_columns_hdf5().items()}

    print_status('[CLIENT]','[Num Sim] Done.')

    new_dict = {
        str(mapping[str(outer_k)]): inner
        for outer_k, inner in result.items()
    }
    return new_dict

def similarity_string_gpu(df_str, columns, batch_size=5000):
    # Partition su una colonna alla volta
    cols = df_str.columns
    list_value_col=[]
    mapping = {}

    for index_col in range(len(cols)):
        col = cols[index_col]
        # embedding_matrix = np.vstack(df_str[col].values)
        parsed = df_str[col].apply(parse_vector).dropna()

        if parsed.empty:
            raise ValueError(f"[CLIENT] Nessun embedding valido trovato in colonna {col}")

        # parsed.values contiene stringhe, convertile in array
        embedding_list = [
            np.array(ast.literal_eval(s)) if isinstance(s, str) else np.array(s) for s
            in parsed.values]

        # embedding_matrix = np.vstack(embedding_list)

        mapping[str(columns.index(col))] = str(col)

        # Step GPU
        res = faiss.StandardGpuResources()

        # Normalizza i vettori per usare la cosine similarity
        embedding_matrix = np.vstack(embedding_list).astype('float32', copy=False)
        faiss.normalize_L2(embedding_matrix)

        # Crea indice (inner product equivale a cosine dopo normalizzazione)
        index_flat = faiss.IndexFlatIP(embedding_matrix.shape[1])
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)

        # Aggiungi gli embeddings all'indice
        gpu_index.add(embedding_matrix)

        max_k = 2048
        total_k = embedding_matrix.shape[0]
        list_value = []

        num_batc = [i for i in range(0, total_k, max_k)]

        pbar = tqdm(desc='[CLIENT] Compute distance for each batch', total=len(num_batc), leave=False, disable=True)
        for start_k in range(0, total_k, max_k):
            end_k = min(start_k + max_k, total_k)
            query_batch = embedding_matrix[start_k:end_k]
            D_part, I_part = gpu_index.search(query_batch, total_k)
            list_value.append((D_part, I_part))
            pbar.update(1)

        list_value_col.append((col, list_value))

    result = {}
    pbar = tqdm(desc='[CLIENT] Compute distance for each column',
                               total=len(list_value_col), leave=False)
    for indice_col_value in range(len(list_value_col)):
        col = list_value_col[indice_col_value][0]
        value = list_value_col[indice_col_value][1]

        thr = float((get_thr()).get(col, {}).get('threshold', 0.0))
        if thr == 1.0:
            low, high = 0.90, 1.0
        else:
            low, high = thr - 0.10, thr + 0.10

        result[str(columns.index(col))] = {}

        # Carica i memmap dei file di distanze e indici
        D_list = [D_val for D_val, _ in value]
        I_list = [I_val for _, I_val in value]

        num_batches = len(D_list)
        batch_size = D_list[0].shape[0]

        for batch_i in range(num_batches):
            for i in range(batch_size):
                query_index = batch_i * max_k + i
                if query_index >= total_k:
                    continue

                d_row = D_list[batch_i][i]
                i_row = I_list[batch_i][i]

                filtered = [(dist, idx) for dist, idx in zip(d_row, i_row)
                            if low <= dist <= high and query_index != idx]

                # Ordina per distanza (più alta prima)
                filtered.sort(key=lambda x: x[0], reverse=True)
                # Converti in lista di stringhe
                valid = set([str(int(idx)) for dist, idx in filtered] + [str(query_index)])

                if valid:
                    result[str(columns.index(col))][str(query_index)] = list(valid)
        # === PULIZIA MEMORIA per questa colonna ===
        del D_list, I_list
        gc.collect()
        pbar.update(1)

    # === PULIZIA FINALE ===
    del df_str
    gc.collect()

    print_status('\n[CLIENT]','[String Sim] Done.')

    new_dict = {
        str(mapping[outer_k]): inner
        for outer_k, inner in result.items()
    }

    return new_dict
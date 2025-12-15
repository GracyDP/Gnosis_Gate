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

@cuda.jit
def gpu_compute_similarity(data, thresh_arr, sim_mat, n_rows, n_cols):
    c = cuda.blockIdx.z
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if c < n_cols and i < n_rows and j < n_rows:
        v_i = data[i * n_cols + c]
        v_j = data[j * n_cols + c]
        thresh = thresh_arr[c]
        sim_mat[c, i, j] = 1 if abs(v_i - v_j) <= thresh else 0

def similarity_numeric_gpu(df_num, original_col):
    thresholds = get_thr()

    cols = df_num.columns

    n_rows, n_cols = df_num.shape[0], len(cols)

    # Prepara array dati solo per colonne numeric
    data = np.zeros((n_rows, n_cols), dtype=np.float32)
    thresh_list = []
    mapping = {}

    pbar = tqdm(desc='[CLIENT] Loading numeric columns', total=n_cols, leave=False)
    for idx, col in enumerate(cols):
        info = thresholds.get(col)
        mapping[str(original_col.index(col))] = str(col)

        if info and info.get('type') == 'numeric' and not info.get('sensitive'):
            data[:, idx] = (df_num[col].to_numpy().astype(np.float32))

        if not info or info.get('type') != 'numeric' or info.get('sensitive'):
            thresh_list.append(0.0)
        else:
            thresh_list.append(float(info.get('threshold', 0)))
        pbar.update(1)

    thresh_arr = np.array(thresh_list, dtype=np.float32)

    # Configurazione blocco/griglia
    dev = cuda.get_current_device()
    max_thr = dev.MAX_THREADS_PER_BLOCK
    side = int(np.floor(np.sqrt(max_thr)))
    bx, by = min(side, n_rows), min(side, n_rows)
    block = (bx, by, 1)
    grid = (ceil(n_rows / bx), ceil(n_rows / by), n_cols)

    # 3) Copia dati su GPU, alloca d_sim direttamente in device
    d_data = cuda.to_device(data.ravel())
    d_thresh = cuda.to_device(thresh_arr)
    d_sim = cuda.device_array((n_cols, n_rows, n_rows), dtype=np.int8)

    # 4) Lancia il kernel
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
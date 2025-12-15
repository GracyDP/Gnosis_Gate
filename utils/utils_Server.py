import base64
import glob
import json
import multiprocessing
import os
import warnings

warnings.filterwarnings("ignore")
from itertools import combinations
from itertools import count
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
import pandas as pd
import torch
from numba import cuda

from utils.Homomorphic.homomorphic_client import convert_from_object_to_ckks, convert_from_base64_to_ckks
from utils.Threshold.threshold_client import get_threshold, get_AllNum_columns, get_AllStr_columns, get_String_NoSens, get_SensCol_from_list, get_StringSens
from utils.utils_common import identifica_tipo, load_columns_from_h5, load_h5_file, print_status

from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import ast
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import gc

batch_size = 5000
batch_update = 1000

base_pth_dataset = ''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def similarita_coseno_embeddings(emb1, emb2, threshold):

    if identifica_tipo(emb1) == 'Embeddings':
        emb1 = np.array(ast.literal_eval(emb1))
    if identifica_tipo(emb2) == 'Embeddings':
        emb2 = np.array(ast.literal_eval(emb2))

    vec1 = np.array(emb1).reshape(1, -1)
    vec2 = np.array(emb2).reshape(1, -1)

    res = cosine_similarity(vec1, vec2)[0][0]

    if threshold == 1.0:
        low, high = 0.90, 1.0
    else:
        low, high = threshold - 0.10, threshold + 0.10
    # Restituisco un risultato in base alla soglia
    # return 1 if res >= threshold else -1
    # Controllo se la similarità rientra nell'intorno
    return 1 if low <= res <= high else -1

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def confronta(val1, val2):
    numeric_types = (int, float, np.integer, np.floating)
    if isinstance(val1, numeric_types) and isinstance(val2, numeric_types):
        if val1 > val2:
            return 1
        elif val1 < val2:
            return -1
        elif val1 == val2:
            return 0
    raise ValueError("Valori non confrontabili direttamente")

def chunk_dict(data, num_chunks, min_chunk_size=2):
    """
    Divide un dizionario in num_chunks parti circa uguali.
    """
    items = list(data.items())

    # Riduci num_chunks se servono chunk troppo piccoli
    while num_chunks > 1 and ceil(len(items) / num_chunks) < min_chunk_size:
        num_chunks -= 1

    chunk_size = ceil(len(items) / num_chunks)
    return [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]

def build_minimal_confronti(mapping):
    """
    Restituisce:
      - minimal: dict {rappresentante_pos: [posizioni_cifrate...]} (una voce per gruppo di posizioni in chiaro)
      - pos_to_rep: dict {posizione: rappresentante_pos} per risalire la voce unica del gruppo
      - pairs: list di tuple (rappresentante_pos, posizione_cifrata) comoda per iterare i confronti
    """
    # separo gruppi in chiaro ed elenchi di posizioni cifrate
    clear_groups = {}
    enc_positions = []
    for k, v in mapping.items():
        # prova a parsare la chiave: se è una lista/tuple, la considero "embedding in chiaro"
        is_clear = False
        try:
            parsed = ast.literal_eval(k)
            if isinstance(parsed, (list, tuple)):
                is_clear = True
        except Exception:
            is_clear = False

        if is_clear:
            clear_groups[k] = list(v)
        else:
            enc_positions.extend(v)

    # de-duplico mantenendo ordine
    enc_positions = list(OrderedDict.fromkeys(enc_positions))
    # costruisco mapping minimale: un rappresentante per gruppo in chiaro
    minimal = {}
    # pos_to_rep = {}
    for key, pos_list in clear_groups.items():
        if not pos_list:
            continue

        rep = tuple(pos_list[0]) if isinstance(pos_list[0], list) else pos_list[0]
        minimal[rep] = enc_positions[:]   # copia della lista cifrata (stessa per ogni gruppo chiaro)

    return minimal

def retrive_metadata_from_h5(path_df):
    # Apri l'HDF5 in modalità read-only
    store = pd.HDFStore(path_df, mode="r")

    # Lista delle chiavi (ognuna corrisponde a una "colonna" nel tuo caso)
    keys = store.keys()

    # Prendiamo la prima chiave e leggiamo solo gli indici per sapere quante righe ha
    first_key = keys[0]
    n_rows = store.get_storer(first_key).nrows

    store.close()
    original_col = [(key.replace('col_','')).replace('/','') for key in keys]

    return original_col, n_rows, len(original_col)

def extract_string(string_dict, mapping, string_no_sens, string_sens):

    righe_sens = set()
    righe = set()
    # Output dictionaries
    new_mapping = defaultdict(list)
    rimossi_ses = defaultdict(list)
    rimossi = defaultdict(list)
    tot = len(list(string_dict.items())) + len(mapping.items())

    pbar = tqdm(total=tot, colour='cyan', desc='[SERVER][Minimum] Extract String', leave=False)
    for col, val in string_dict.items():
        if ',' in val and val.endswith(',_'):
            riga = val.split(',')[0]
            if col in string_no_sens:
                righe.add(riga)
            elif col in string_sens:
                righe_sens.add(riga)
        pbar.update(1)

    # Scorri mapping e filtra
    for k, indici in mapping.items():
        da_tenere = [i for i in indici if i.split(',')[0] not in righe and i.split(',')[0] not in righe_sens]
        da_rimuovere = [i for i in indici if i.split(',')[0] in righe and i.split(',')[0] not in righe_sens]
        da_rimuovere_sens = [i for i in indici if i.split(',')[0] in righe_sens and i.split(',')[0] not in righe]

        if da_tenere:
            new_mapping[k].extend(da_tenere)
        if da_rimuovere:
            rimossi[k].extend(da_rimuovere)
        if da_rimuovere_sens:
            rimossi_ses[k].extend(da_rimuovere_sens)

        pbar.update(1)

    return rimossi, rimossi_ses, new_mapping

def build_groups(stringNoSens, dict_col):
    # Inizializzo new_mapping con embedding come chiavi
    new_mapping = {}
    if not stringNoSens:
        return new_mapping

    # Preprocessing
    parsed_keys = {}
    idx_to_col = {idx.split(",")[0]: name for name, idx in dict_col.items()}

    for k, v in stringNoSens.items():
        tipo = identifica_tipo(k)
        if tipo == "Embeddings":
            parsed_keys[k] = np.array(ast.literal_eval(k))
        elif tipo == "ndarray":
            parsed_keys[k] = np.array(k)  # se già ndarray
        else:
            raise Exception('Tipo chiave sconosciuto')
        new_mapping[str(list(k)) if tipo == "ndarray" else k] = v

    all_pairs = list(combinations(stringNoSens.items(), 2))
    pbar = tqdm(total=len(all_pairs), desc="[SERVER][Minimum] Build Group String", colour='green', leave=False)
    update_counter = 0

    for (candidate_key, candidate_val), (target_key, target_val) in all_pairs:
        tipo_c = identifica_tipo(candidate_key)
        pos_i = stringNoSens[candidate_key][0] if tipo_c in ("Embeddings", "ndarray") else None
        col_idx = pos_i.split(",")[0] if pos_i else None
        col_name = idx_to_col.get(col_idx)
        if col_name is None:
            update_counter += 1
            if update_counter >= batch_update:
                pbar.update(update_counter)
                update_counter = 0
            continue  # se non trovo la colonna, salto

        candidate_np = parsed_keys[candidate_key]
        candidate_emb = str(list(candidate_key)) if tipo_c == "ndarray" else candidate_key

        tipo_t = identifica_tipo(target_key)
        if tipo_t not in ("Embeddings", "ndarray"):
            raise Exception('Tipo target sconosciuto')
        target_np = parsed_keys[target_key]
        target_emb = str(list(target_key)) if tipo_t == "ndarray" else target_key

        if candidate_emb != target_emb:
            if similarita_coseno_embeddings(candidate_np, target_np, float(get_threshold(col_name))) == 1:
                new_mapping[candidate_emb].extend(new_mapping[target_emb])
                new_mapping.pop(target_emb, None)

        update_counter += 1
        if update_counter >= batch_update:
            pbar.update(update_counter)
            update_counter = 0

    pbar.close()
    return new_mapping

def find_key_by_value(d, target):
    for k, v in d.items():
        if target in v:
            return k
    return None

def comparison_string(mapping_string, stringSens, stringNoSens, string_col):
    """
    Versione batch di comparison_string.
    Logica invariata: valori inviati al server sono i risultati di cosine_similarity_enc.
    thr: soglia unica per tutti i target.
    """
    print(f'Comparison string dentro')
    minimal = build_minimal_confronti(mapping_string)
    total_comparisons = sum(len(v) for v in minimal.values())
    if total_comparisons != 0:
        pbar = tqdm(total=total_comparisons, desc="[SERVER][Minimum] Comparison String", colour='blue', leave=False)
        update_counter = 0

        while minimal:
            candidate = next((v for v in minimal if identifica_tipo(v) != 'CKKSVector'), None)
            if candidate is None:
                candidate = next(iter(minimal))

            embed_candidate_key = find_key_by_value(mapping_string, candidate)
            tipo_candidate = identifica_tipo(embed_candidate_key)
            da_aggiungere = []

            # Preparazione candidato
            if tipo_candidate == "CKKSVector":
                candidate_ckks = embed_candidate_key
                candidate_np = candidate_emb = None
                pos_i = stringSens[embed_candidate_key][0]
            else:
                candidate_ckks = None
                candidate_np = (
                    np.array(ast.literal_eval(embed_candidate_key))
                    if tipo_candidate == "Embeddings"
                    else embed_candidate_key
                )
                candidate_emb = embed_candidate_key
                pos_i = stringNoSens[embed_candidate_key][0]  # es: "3,0"

            batch_ckks = []
            batch_keys = []

            col_idx = pos_i.split(",")[0]  # "3"

            # Trovo il nome della colonna corrispondente
            col_name = None
            for name, idx in string_col.items():
                if idx.startswith(col_idx + ","):
                    col_name = name
                    break
            if col_name is None:
                continue  # se non trovo la colonna, salto
            thr = get_threshold(col_name)
            print('Candidate: ', embed_candidate_key[:10] if isinstance(embed_candidate_key, str) else embed_candidate_key)
            print('minimal[candidate]: ', minimal[candidate])
            for target_pos in minimal[candidate]:
                print('target_pos: ', target_pos)
                exit(0)
                embed_target_key = find_key_by_value(mapping_string, target_pos)
                tipo_target = identifica_tipo(embed_target_key)

                # Calcola cosine_similarity_enc solo se necessario
                if tipo_target in ("Embeddings", "ndarray") and tipo_candidate == "CKKSVector":
                    target_np = (
                        np.array(ast.literal_eval(embed_target_key))
                        if tipo_candidate == "Embeddings"
                        else embed_target_key
                    )

                    #Similarità coseno -> essendo gia normalizzati, si riduce a val1 * val 2.
                    res_enc = candidate_ckks.dot(convert_from_object_to_ckks(normalize(target_np)))
                elif tipo_target == "CKKSVector" and tipo_candidate in ("Embeddings", "ndarray"):
                    candidate_enc = convert_from_object_to_ckks(normalize(candidate_np))
                    res_enc = candidate_enc.dot(embed_target_key)
                elif tipo_target == "CKKSVector" and tipo_candidate == "CKKSVector":
                    res_enc = candidate_ckks.dot(embed_target_key)
                else:
                    update_counter += 1
                    if update_counter >= batch_update:
                        pbar.update(update_counter)
                        update_counter = 0
                    continue

                batch_ckks.append(res_enc)
                batch_keys.append(embed_target_key)

                # Invia batch al server
                if len(batch_ckks) >= batch_size:
                    da_aggiungere.extend(send_embeddings_batch(batch_ckks, thr, batch_keys))
                    update_counter += len(batch_ckks)
                    if update_counter >= batch_update:
                        pbar.update(update_counter)
                        update_counter = 0
                    batch_ckks.clear()
                    batch_keys.clear()

            # Invio eventuale batch rimanente
            if batch_ckks:
                da_aggiungere.extend(send_embeddings_batch(batch_ckks, thr, batch_keys))
                update_counter += len(batch_ckks)
                if update_counter >= batch_update:
                    pbar.update(update_counter)
                    update_counter = 0

            # Merge dei mapping (logica invariata)
            minimal.pop(candidate)
            seen = set(mapping_string[embed_candidate_key])
            for key in da_aggiungere:
                for val_target in mapping_string[key]:
                    if val_target not in seen:
                        seen.add(val_target)
            mapping_string[embed_candidate_key] = list(seen)
            for key in da_aggiungere:
                mapping_string.pop(key)

        if update_counter > 0:
            pbar.update(update_counter)
        pbar.close()
    return mapping_string

def assign_rank_string(mapping_string, current_rank):
    ranking_string = {}

    # Iterazione su mapping_string
    pbar = tqdm(total=len(mapping_string), desc="[SERVER][Minimum] Add Rank String", colour='yellow', leave=False)
    update_counter = 0
    for key, value in mapping_string.items():

        ranking_string[current_rank] = value
        current_rank += 1
        update_counter += 1

        if update_counter >= batch_update:
            pbar.update(update_counter)
            update_counter = 0

    return ranking_string

def salva_json(dati, percorso_file):
    """
    Salva un oggetto Python (di solito un dizionario o una lista) in un file JSON.

    Parametri:
    - dati: l'oggetto Python da salvare (dict, list, ecc.)
    - percorso_file: il percorso completo del file JSON da creare o sovrascrivere
    - indent: il livello di indentazione per una migliore leggibilità (default = 4)
    """
    try:
        with open(percorso_file, 'w') as f:
            json.dump(dati, f,  indent=4)
    except Exception as e:
        raise Exception(f"Errore durante il salvataggio del file JSON: {e}")

def replace_df(merged, path_df):

    total_cells = sum(len(pos_list) for pos_list in merged.values())
    update = 0
    df = load_h5_file(path_df)
    with tqdm(total=total_cells, desc='[SERVER][Matrix] Creating', leave=False, colour='cyan') as pbar:
        for rank, positions in merged.items():
            for i, pos in enumerate(positions):
                try:
                    col_idx, row_idx = map(int, pos.split(','))
                    df.iat[row_idx, col_idx] = rank
                    update = update + 1
                    if update == batch_update:
                        pbar.update(update)
                        update = 0
                except Exception as e:
                    raise Exception(f'Posizione(col_idx, row_idx) {pos} non trovata nel df.')

    del merged
    gc.collect()
    return df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

def export_columns_to_h5_streaming(d_sim, output_path='tmp_file.h5'):
    """
    Process GPU similarity slices sequentially and save each column's matches to a separate JSON Lines file
    on the fly, without building the entire column dict in memory.

    Args:
        d_sim: GPU device array of shape (n_cols, H, W)
        batch_size: Number of records to buffer before writing to file (for performance)
    """
    n_cols, H, W = d_sim.shape
    buffer_shape = (H, W)

    # Se esiste, lo sovrascrive
    if os.path.exists(output_path):
        os.remove(output_path)

    pbar = tqdm(range(n_cols), desc="[SERVER] Streaming columns", leave=True, disable=True)
    with pd.HDFStore(output_path, mode='w') as store:

        for ci in range(n_cols):
            host_buf = cuda.pinned_array(buffer_shape, dtype=np.int8)
            d_sim[ci].copy_to_host(host_buf)

            i_idx, j_idx = np.where(host_buf == 1)

            # Costruisci DataFrame e Salva con chiave col_{ci}
            store.put(f'col_{ci}', pd.DataFrame({'i': i_idx, 'j': j_idx}), format='table')  # 'table' = supporta query + append

            # Cleanup
            del host_buf, i_idx, j_idx
            gc.collect()

            pbar.update(1)

    print_status('\n[CLIENT]',f"[Similarity] All {n_cols} columns streamed to H5 files!")

def load_all_columns_hdf5(h5_path='tmp_file.h5', delete_file=True):
    """
    Carica tutte le colonne da un file HDF5 e le aggrega in un dizionario annidato:
        result[ci][i] = [j1, j2, ...]

    Args:
        h5_path: percorso al file .h5 contenente le chiavi 'col_{ci}'
        delete_file: se True, elimina il file .h5 dopo il caricamento

    Returns:
        result: dict[str, dict[str, list[str]]]
    """
    result = {}

    with pd.HDFStore(h5_path, mode='r') as store:
        for key in store.keys():  # keys like '/col_0', '/col_1', ...
            ci = key.strip('/').split('_')[1]  # 'col_42' → '42'
            df = store[key]

            col_dict = {}
            for i, j in zip(df['i'], df['j']):
                key_i = str(i)
                key_j = str(j)
                col_dict.setdefault(key_i, []).append(key_j)

            result[ci] = col_dict

    if delete_file:
        os.remove(h5_path)

    return result

def load_all_columns_jsonl(folder='.', pattern='col_*.jsonl'):
    """
    Load and concatenate all per-column JSON Lines files into a single nested dict.

    Returns:
        result: dict mapping column index (str) to dict mapping row i (str) to list of matched j (str)
    """
    result = {}
    filepaths = glob.glob(os.path.join(folder, pattern))
    for path in filepaths:
        basename = os.path.basename(path)
        ci = basename.split('_')[1].split('.')[0]
        col_dict = {}
        with open(path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                key_i = str(rec['i'])
                key_j = str(rec['j'])
                col_dict.setdefault(key_i, []).append(key_j)

        result[ci] = col_dict

    for p in filepaths:
        os.remove(p)

    return result

batch_counter = count()

def flush_batch(batch, useMongoDB, tmp_dir, col, tipo):
    """
    Scrive batch su MongoDB o file temporaneo
    """
    for row in batch:
        row[tipo] = base64.b64encode(row[tipo].serialize()).decode("utf-8")
    else:
        idx = next(batch_counter)
        path = f"{tmp_dir}{col}_batch_{idx}.parquet"
        df = pd.DataFrame(batch)
        df.to_parquet(path, index=False)

    del batch, df
    
def merge_column_batches(tmp_dir, col_name, final_file):
    """
    Ricombina tutti i file batch HDF5 di una colonna in un unico file HDF5.

    Args:
        tmp_dir (str): cartella dove sono salvati i batch.
        col_name (str): nome della colonna da ricombinare.
        final_file (str): percorso del file HDF5 finale.

    Returns:
        None
    """
    # Recupera tutti i file batch della colonna
    pattern = tmp_dir + f"{col_name}_batch_*.parquet"
    batch_files = glob.glob(pattern)

    if not batch_files:
        raise Exception(f"Nessun file batch trovato per la colonna '{col_name}'.")

    batch_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))

    # Leggi e concatena tutti i batch verticalmente e Salva in un unico HDF5
    # (pd.concat([pd.read_hdf(f, key=col_name) for f in batch_files], ignore_index=True)).to_hdf(final_file, key=col_name, mode='w', format='table', data_columns=True)

    # Leggi e concatena tutti i batch verticalmente e Salva in un unico parquet
    (pd.concat([pd.read_parquet(f) for f in batch_files], ignore_index=True)).to_parquet(final_file,index=False)

    # Rimuovi i file batch temporanei
    for f in batch_files:
        os.remove(f)
"""
====================================================
Python Server per l’elaborazione sicura di dataset
====================================================
Questo server Flask espone varie API per:
- Salvare e gestire soglie di configurazione (`/save_thr`)
- Caricare un contesto TenSEAL per la crittografia omomorfica (`/upload_context`)
- Caricare e processare dataset da file (`/`)
- Calcolare partizioni e similarità tra dati sensibili (Crittografati con schema CKKS)
- Valutare colonne e costruire una matrice di vincoli DC (`/evaluate-column`)
- Chiudere la connessione e liberare memoria (`/end_connection`)

Funzionalità principali:
- Supporta input in diversi formati (HDF5, Parquet, CSV)
- Utilizza TenSEAL per gestire dati crittografati (lazy vectors)
- Salva log delle performance con memory_profiler
- Genera file intermedi (HDF5, CSV, JSON) per l’elaborazione
"""

import gc
import logging
import os
import warnings

#3
import json
import ast
import numpy as np
#**********NEW
import traceback
#**********END
#

import pandas as pd
import sys, tenseal as ts
from flask import Flask, request, jsonify

from utils.utils_Server import salva_json
from utils.utils_common import save_df_h5, print_status, the_end

print("[CLIENT] python:", sys.executable)
print("[CLIENT] tenseal:", ts.__version__)


warnings.filterwarnings("ignore")

app = Flask('python-server')

json_item = {}
context = None

path = ''

fx = open("log/log_server.txt","w")


#**********NEW
def _infer_modality_from_path(path: str):
    """Deduce modality from experiment path if present."""
    p = (path or "").replace("\\", "/")
    for m in ["Manual", "All_NOT_Sensitive", "All_Sensitive"]:
        if f"/{m}/" in p:
            return m
    return None


def _get_h5_colnames(path_df: str):
    """
    Caso A (atteso): HDF5 con keys /col_<name>
    Caso B (fallback): HDF5 con una singola tabella (prende df.columns dalla prima key)
    Ritorna: (colnames, table_key_fallback)
    """
    with pd.HDFStore(path_df, mode="r") as store:
        keys = store.keys()

    # DEBUG: stampo le keys disponibili
    print("[SERVER][DEBUG] H5 keys:", keys)

    # Caso A: /col_<name>
    colnames = sorted([k[len("/col_"):] for k in keys if k.startswith("/col_")])
    if colnames:
        return colnames, None

    # Caso B: tabella unica
    if not keys:
        return [], None

    table_key = keys[0]
    try:
        df = pd.read_hdf(path_df, key=table_key)
        return list(df.columns), table_key
    except Exception:
        return [], None


def _build_numeric_sim(s: pd.Series, thr: float) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    diff = np.abs(x[:, None] - x[None, :])
    return (diff <= float(thr)).astype(np.int8)


def _build_vector_cosine_sim(s: pd.Series, thr: float) -> np.ndarray:
    """
    Expects vectors stored as:
    - list/tuple/np.ndarray OR
    - string like "[0.1, 0.2, ...]"
    If not parsable, uses a zero-vector fallback (won’t crash, but similarity may be meaningless).
    """
    vecs = []
    for v in s.tolist():
        if isinstance(v, (list, tuple, np.ndarray)):
            vecs.append(np.array(v, dtype=np.float32))
        elif isinstance(v, str) and v.strip().startswith("[") and v.strip().endswith("]"):
            try:
                vecs.append(np.array(ast.literal_eval(v), dtype=np.float32))
            except Exception:
                vecs.append(np.zeros(1, dtype=np.float32))
        else:
            vecs.append(np.zeros(1, dtype=np.float32))

    dims = [len(v) for v in vecs if len(v) > 0]
    d = min(dims) if dims else 1
    X = np.stack([v[:d] if len(v) >= d else np.pad(v, (0, d-len(v))) for v in vecs], axis=0)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T
    return (sim >= float(thr)).astype(np.int8)
#**********END


@app.route("/save_thr", methods=["POST"])
def save_thr():
    """
        Endpoint per salvare la soglia (thr) inviata dal client.
        - Riceve JSON {"thr": {...}, "path": "..."}
        - Salva il contenuto in memoria
        - Stampa log e restituisce conferma
        """

    global json_item
    print_status('[SERVER]','[Save thr] Server connected.')
    data = request.get_json(force=True)

    # Estraggo il dict thr e il path
    json_item = data.get("thr")
    global path
    path = data.get("path")

    # Controllo se il campo thr è presente
    if json_item is None:
        print_status('[SERVER]','[Save thr] Error: Threshold not found.')
        return jsonify({"error": "Nessun campo 'thr' nel payload"}), 400
    else:
        print_status('[SERVER]','[Save thr] Threshold file found.')
        return jsonify({"status": "saved"}), 200


@app.route("/upload_context", methods=["POST"])
@app.route("/upload_context", methods=["POST"])
def upload_context():
    """
    Endpoint per caricare un contesto TenSEAL inviato come file.
    - Legge il contesto crittografico dal client
    - Lo memorizza in una variabile globale
    """

    global context
    raw = request.files["file"].read()

    # Stampa la lunghezza dei byte ricevuti
    print("[SERVER] Upload_context: lunghezza bytes ricevuti:", len(raw))

    context = ts.context_from(raw)

    print_status('[SERVER]','[CKKS] Context file found.')
    return "Contesto pubblico ricevuto", 200


@app.route("/end_connection", methods=["POST"])
def end_connection():
    """
    Endpoint per chiudere la connessione:
    - Resetta le variabili globali
    - Libera memoria con gc.collect()
    """
    werk_log = logging.getLogger('werkzeug')
    prev_level = werk_log.level
    werk_log.setLevel(logging.ERROR)
    try:
        flag = request.form.get("flag")
        flag = flag.lower() == 'true' if flag is not None else False

        path, context, json_item = None, None, None
        gc.collect()
        print_status('[SERVER]','End Connection.')
        os.system('cls' if os.name == 'nt' else 'clear')
        if flag:
            the_end('[SERVER]')

        return "Connection Closed!", 200
    finally:
        werk_log.setLevel(prev_level)


@app.route("/evaluate-column", methods=["GET"])
def evaluate_column():
    """
    Endpoint per valutare le colonne del dataset.
    Passaggi:
    1) Carica dataset (HDF5, Parquet, CSV)
    2) Applica mapping e genera matrice di DC
    3) Salva i risultati su file (CSV + JSON)
    4) Restituisce conferma
    """
    data = request.get_json()

    path_df = data['path_df']
    path = data['path']

    print_status('[SERVER]','[Minimum][Dataframe] Loading documents.')
    # Caricamento dataset in base all’estensione
    if path_df:
        if '.h5' in path_df:
            pass  # Nessuna azione, già in formato corretto

        elif '.parquet' in path_df:
            try:
                path_df = path_df.replace('.parquet', '.h5')
                save_df_h5(df_path=path_df, df=pd.read_parquet(path_df))
            except Exception as e:
                return {'error': str(e)}, 500

        elif '.csv' in path_df:
            try:
                path_df = path_df.replace('.csv', '.h5')
                save_df_h5(df_path=path_df, df=pd.read_parquet(path_df))  # incompatibile con csv (come nel tuo file)
            except Exception as e:
                return {'error': str(e)}, 500

        else:
            raise Exception("[Minimum][Dataframe] Extension of DB Not found.")
    else:
        raise Exception("[Minimum][Evaluate column] Dataframe Not found or is empty.")

    # Avvio elaborazione colonne
    print_status('[SERVER]','[Minimum][Evaluate column] Start.')

    print('Computazione Matrice dei Minimi TO DO')

    #**********NEW
    # ============================
    # COMPUTAZIONE MATRICE SIMILARITÀ (per colonna) + matrice aggregata
    # ============================
    global json_item
    if not json_item:
        return jsonify({"error": "Thresholds non presenti (chiama /save_thr prima)."}), 400

    modality = _infer_modality_from_path(path)

    # --- Leggo le colonne dall'HDF5 (supporto /col_* oppure tabella unica) ---
    try:
        colnames, table_key_fallback = _get_h5_colnames(path_df)
        if not colnames:
            return jsonify({"error": f"Nessuna colonna trovata in HDF5: {path_df}"}), 500
    except Exception as e:
        print("[SERVER][ERROR] Errore durante lettura HDF5/keys:", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    # --- Selezione colonne in base a modality + thresholds ---
    def should_use_col(col: str) -> bool:
        info = json_item.get(col)
        if info is None:
            return False

        if modality == "All_NOT_Sensitive":
            return info.get("sensitive", False) is False

        if modality == "All_Sensitive":
            return True

        # Manual (fallback): includo tutto ciò che ha thr
        return True

    selected_cols = [c for c in colnames if should_use_col(c)]
    if not selected_cols:
        return jsonify({"error": f"Nessuna colonna selezionata (modality={modality})."}), 400

    # Se siamo in fallback tabella unica, leggiamo la tabella una sola volta
    df_all_fallback = None
    if table_key_fallback is not None:
        try:
            df_all_fallback = pd.read_hdf(path_df, key=table_key_fallback)
        except Exception as e:
            print("[SERVER][ERROR] Errore lettura tabella fallback:", e)
            print(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    # --- Carico tutte le Series selezionate ---
    series_map = {}
    n_rows = None
    for c in selected_cols:
        try:
            if table_key_fallback is None:
                s = pd.read_hdf(path_df, key=f"/col_{c}")
            else:
                s = df_all_fallback[c]
        except Exception as e:
            print(f"[SERVER][ERROR] Lettura colonna fallita: {c} - {e}")
            print(traceback.format_exc())
            return jsonify({"error": f"Errore lettura colonna {c}: {e}"}), 500

        if n_rows is None:
            n_rows = len(s)
        else:
            if len(s) != n_rows:
                return jsonify({"error": f"Colonna {c} ha lunghezza diversa ({len(s)} vs {n_rows})."}), 500

        series_map[c] = s

        # DEBUG PRINT: contenuto reale colonne (1 valore di esempio)
        try:
            sample_val = s.iloc[0] if hasattr(s, "iloc") else (s[0] if len(s) else None)
            print(
                f"[SERVER][DEBUG] COL={c} | "
                f"thr_type={json_item.get(c, {}).get('type')} | "
                f"py_type={type(sample_val)} | "
                f"value_preview={str(sample_val)[:80]}"
            )
        except Exception as e:
            print(f"[SERVER][DEBUG] COL={c} | ERRORE lettura sample: {e}")

    # --- Output dirs ---
    out_dir = os.path.join(path, "Matrices")
    os.makedirs(out_dir, exist_ok=True)

    produced = []
    skipped = []

    # matrice aggregata (conta quante colonne “simili” per ogni coppia)
    agg = np.zeros((n_rows, n_rows), dtype=np.int16)

    for c in selected_cols:
        info = json_item.get(c, {})
        ctype = info.get("type", "numeric")
        cthr = info.get("threshold", 0)

        try:
            if ctype == "numeric":
                M = _build_numeric_sim(series_map[c], cthr)
            else:
                M = _build_vector_cosine_sim(series_map[c], cthr)

            out_csv = os.path.join(out_dir, f"sim_{c}.csv")
            pd.DataFrame(M).to_csv(out_csv, index=False)

            agg += M.astype(np.int16)
            produced.append(c)

        except Exception as e:
            skipped.append({"col": c, "error": str(e)})

    # Salvo matrice aggregata
    matrix_agg_path = os.path.join(path, "Matrix_DCs.csv")
    pd.DataFrame(agg).to_csv(matrix_agg_path, index=False)

    # Salvo summary/mapping
    summary = {
        "path_df": path_df,
        "modality": modality,
        "n_rows": n_rows,
        "columns_found": colnames,
        "columns_selected": selected_cols,
        "columns_produced": produced,
        "columns_skipped": skipped,
        "matrices_dir": out_dir,
        "matrix_aggregated": matrix_agg_path,
        "table_key_fallback": table_key_fallback,
    }
    salva_json(dati=summary, percorso_file=os.path.join(path, "Rank_map.json"))

    #FINE MATRICE DEI MINIMI
    #**********END

    print_status('[SERVER]','[Minimum][Evaluate column] Done.')

    return jsonify({
        "status": "DONE",
        "message": "Dati processati con successo",
    }), 200


# Avvio del server Flask
app.run(port=5000, threaded=True, use_reloader=False)

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

import pandas as pd
import tenseal as ts
from flask import Flask, request, jsonify

from utils.utils_Server import salva_json
from utils.utils_common import save_df_h5, print_status, the_end

warnings.filterwarnings("ignore")

app = Flask('python-server')

json_item = {}
context = None

path = ''

fx = open("log/log_server.txt","w")

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
def upload_context():
    """
    Endpoint per caricare un contesto TenSEAL inviato come file.
    - Legge il contesto crittografico dal client
    - Lo memorizza in una variabile globale
    """

    global context
    context = ts.context_from(request.files["file"].read())

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
                pass # Nessuna azione, già in formato corretto

        elif '.parquet' in path_df:
            try:
                path_df = path_df.replace('.parquet', '.h5')
                save_df_h5(df_path=path_df, df=pd.read_parquet(path_df))
            except Exception as e:
                return {'error': str(e)}, 500

        elif '.csv' in path_df:
            try:
                path_df = path_df.replace('.csv', '.h5')
                save_df_h5(df_path=path_df, df=pd.read_parquet(path_df))
            except Exception as e:
                return {'error': str(e)}, 500

        else:
            raise Exception("[Minimum][Dataframe] Extension of DB Not found.")
    else:
        raise Exception("[Minimum][Evaluate column] Dataframe Not found or is empty.")

    # Avvio elaborazione colonne
    print_status('[SERVER]','[Minimum][Evaluate column] Start.')

    print('Computazione Matrice dei Minimi TO DO')

    print_status('[SERVER]','[Minimum][Evaluate column] Done.')

    # Salvataggio risultati

    # Salvo matrice dei minimi in formato df
    # df_all.to_csv(os.path.join(path, 'Matrix_DCs.csv'), index=False)

    # Salvo matrice dei minimi in formato json
    # salva_json(dati=merged, percorso_file=os.path.join(path, 'Rank_map.json'))

    # Restituisce conferma
    return jsonify({
        "status": "DONE",
        "message": "Dati processati con successo",
    }), 200


# Avvio del server Flask
app.run(port=5000, threaded=True, use_reloader=False)
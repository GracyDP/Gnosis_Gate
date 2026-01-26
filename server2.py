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
import numpy as np
import pandas as pd
import tenseal as ts
import h5py
from flask import Flask, request, jsonify
from typing import Dict, List, Tuple, Optional

from utils.utils_Server import salva_json
from utils.utils_common import save_df_h5, print_status, the_end, load_columns_from_h5
from utils.utils_Server_request import request_operation_to_client

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

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFrameLoader:
    """Gestisce il caricamento e conversione di diversi formati di dataset"""
    
    @staticmethod
    def load_and_convert(path_df: str) -> Tuple[str, pd.DataFrame]:
        """
        Carica un dataframe da vari formati e lo converte in HDF5 se necessario.
        
        Args:
            path_df: Path al file del dataframe
            
        Returns:
            Tupla (path_h5, dataframe)
        """
        if not os.path.exists(path_df):
            raise FileNotFoundError(f"File non trovato: {path_df}")
        
        ext = os.path.splitext(path_df)[1].lower()
        
        try:
            if ext == '.h5':
                logger.info(f"Caricamento HDF5: {path_df}")
                return path_df, DataFrameLoader._load_from_h5(path_df)
            
            elif ext == '.parquet':
                logger.info(f"Conversione Parquet -> HDF5: {path_df}")
                df = pd.read_parquet(path_df)
                h5_path = path_df.replace('.parquet', '.h5')
                save_df_h5(df_path=h5_path, df=df)
                return h5_path, df
            
            elif ext == '.csv':
                logger.info(f"Conversione CSV -> HDF5: {path_df}")
                df = pd.read_csv(path_df)  # FIX: era pd.read_parquet
                h5_path = path_df.replace('.csv', '.h5')
                save_df_h5(df_path=h5_path, df=df)
                return h5_path, df
            
            else:
                raise ValueError(f"Estensione non supportata: {ext}")
                
        except Exception as e:
            logger.error(f"Errore nel caricamento del file: {e}")
            raise
    
    @staticmethod
    def _load_from_h5(path: str) -> pd.DataFrame:
        """Carica dataframe da file HDF5"""
        print_status('[SERVER]', f'🔍 DEBUG: Caricamento file H5: {path}')
        
        # Prima prova a leggere un oggetto DataFrame salvato come '/df'
        with h5py.File(path, 'r') as h5file:
            if 'df' in h5file:
                print_status('[SERVER]', f'🔍 DEBUG: Trovato /df, caricamento con pd.read_hdf')
                return pd.read_hdf(path, key='/df')

        # Se non esiste '/df', ricostruisci il DataFrame leggendo tutte
        # le chiavi disponibili nello store (supporta sia 'col_<name>'
        # che chiavi plain come '<name>'). Questo evita di restituire
        # solo la prima colonna quando il file contiene dataset separati.
        with pd.HDFStore(path, mode='r') as store:
            keys = [k.strip('/') for k in store.keys()]
            print_status('[SERVER]', f'🔍 DEBUG: Chiavi trovate: {keys[:5]}...')  # Prime 5

            # Se trovi chiavi con prefisso 'col_' usa la funzione esistente
            col_keys = [k for k in keys if k.startswith('col_')]
            if col_keys:
                columns = [k.replace('col_', '') for k in col_keys]
                print_status('[SERVER]', f'🔍 DEBUG: Uso load_columns_from_h5 per {len(columns)} colonne')
                return load_columns_from_h5(path, columns)

            # Altrimenti concatena tutti i dataset presenti nello store
            if keys:
                df_parts = []
                for idx, raw_key in enumerate(store.keys()):
                    part = store[raw_key]
                    
                    # DEBUG: Mostra tipo del primo valore della prima colonna
                    if idx == 0 and len(part) > 0:
                        first_col = part.columns[0]
                        first_val = part[first_col].iloc[0]
                        print_status('[SERVER]', f'🔍 DEBUG: Prima chiave "{raw_key}"')
                        print_status('[SERVER]', f'         Tipo prima colonna: {type(part[first_col].iloc[0]).__name__}')
                        print_status('[SERVER]', f'         Dtype: {part[first_col].dtype}')
                        print_status('[SERVER]', f'         Ha .serialize? {hasattr(first_val, "serialize")}')
                        print_status('[SERVER]', f'         È bytes? {isinstance(first_val, bytes)}')
                    
                    df_parts.append(part)

                df_full = pd.concat(df_parts, axis=1)

                # Normalizza eventuali prefissi nelle colonne (es. 'col_')
                new_cols = []
                for c in df_full.columns:
                    if isinstance(c, str) and c.startswith('col_'):
                        new_cols.append(c.replace('col_', ''))
                    else:
                        new_cols.append(c)
                df_full.columns = new_cols
                
                print_status('[SERVER]', f'🔍 DEBUG: DataFrame finale shape={df_full.shape}')
                return df_full

            raise ValueError("Nessuna chiave valida trovata nel file HDF5")


class DCMatrixBuilder:
    """
    Costruisce la matrice dei vincoli di denial (Denial Constraints).
    """
    
    def __init__(self, df: pd.DataFrame, thresholds: Dict = None):
        self.df = df
        self.columns = df.columns.tolist()
        self.n_cols = len(self.columns)
        self.thresholds = thresholds or {}
        # Preprocessa i dati una volta sola
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Prepara i dati per l'analisi, gestendo vari formati e dati cifrati"""
        self.col_info = {}
        
        print_status('[SERVER]', f'Analisi di {len(self.columns)} colonne...')
        print_status('[SERVER]', f'Threshold disponibili: {len(self.thresholds)} colonne')
        
        for col in self.columns:
            series = self.df[col].dropna()
            
            # Controlla se la colonna è marcata come sensibile nei threshold
            col_threshold = self.thresholds.get(col, {})
            is_sensitive = col_threshold.get('sensitive', False)
            
            # Debug: mostra info colonna
            if len(series) > 0:
                first_val = series.iloc[0]
                val_type = type(first_val).__name__
                print_status('[SERVER]', f'  Colonna "{col}": tipo={val_type}, sensitive={is_sensitive}, len={len(series)}')
            
            if is_sensitive:
                # Colonna cifrata (sensibile): memorizza info minime
                print_status('[SERVER]', f'Colonna CIFRATA rilevata: {col} (da threshold)')
                self.col_info[col] = {
                    'original': series,
                    'is_encrypted': True,
                    'is_numeric': None,  # Non possiamo saperlo senza decriptare
                    'unique_values': None,
                    'nunique': None,
                    'length': len(series)
                }
            else:
                # Colonna in chiaro: analisi normale
                numeric_series = pd.to_numeric(series, errors='coerce')
                
                self.col_info[col] = {
                    'original': series,
                    'is_encrypted': False,
                    'numeric': numeric_series if numeric_series.notna().sum() > len(series) * 0.5 else None,
                    'is_numeric': pd.api.types.is_numeric_dtype(series) or numeric_series.notna().sum() > len(series) * 0.5,
                    'unique_values': set(series.unique()) if len(series) < 10000 else None,
                    'nunique': series.nunique(),
                    'length': len(series)
                }
    
    def build_predicate_matrix(self) -> Tuple[pd.DataFrame, Dict]:
        """Costruisce la matrice dei predicati per Denial Constraints."""
        logger.info("Costruzione matrice DC...")
        
        matrix = np.zeros((self.n_cols, self.n_cols))
        rank_mapping = {}
        
        for i, col1 in enumerate(self.columns):
            for j in range(i, self.n_cols):
                col2 = self.columns[j]
                
                if i == j:
                    metric = self._compute_diagonal_metric(col1)
                else:
                    metric = self._compute_pairwise_metric(col1, col2)
                
                matrix[i, j] = metric
                matrix[j, i] = metric
                
                rank_mapping[f"{col1}_{col2}"] = {
                    "col1": col1,
                    "col2": col2,
                    "metric_value": float(metric),
                    "rank": i * self.n_cols + j,
                    "col1_index": i,
                    "col2_index": j
                }
        
        df_matrix = pd.DataFrame(matrix, index=self.columns, columns=self.columns)
        
        logger.info(f"Matrice DC costruita: {df_matrix.shape}")
        logger.info(f"Valori non-zero fuori diagonale: {np.count_nonzero(matrix - np.diag(np.diag(matrix)))}")
        
        return df_matrix, rank_mapping
    
    def _compute_diagonal_metric(self, col: str) -> float:
        """Calcola metrica per la diagonale."""
        info = self.col_info[col]
        
        # Se è cifrata, non possiamo calcolare metrica - ritorna 1.0 come placeholder
        if info.get('is_encrypted'):
            return 1.0
        
        if info['length'] == 0:
            return 0.0
        
        if info['is_numeric'] and info['numeric'] is not None:
            series = info['numeric'].dropna()
            if len(series) > 0:
                min_val, max_val = series.min(), series.max()
                if max_val != min_val:
                    return float(max_val - min_val)
        
        # Fallback: cardinalità normalizzata
        return float(info['nunique']) / info['length']
    
    def _compute_pairwise_metric(self, col1: str, col2: str) -> float:
        """Calcola metrica di relazione tra due colonne diverse."""
        info1 = self.col_info[col1]
        info2 = self.col_info[col2]
        
        # Se ALMENO UNA colonna è cifrata, richiedi al client
        if info1.get('is_encrypted') or info2.get('is_encrypted'):
            print_status('[SERVER]', f'🔐 Rilevati dati cifrati: {col1} x {col2}')
            try:
                # Richiedi al client di calcolare la correlazione/similarità
                # Passa i nomi delle colonne - il client le ha in memoria
                result = request_operation_to_client(
                    operation="correlation",
                    col1=col1,
                    col2=col2
                )
                print_status('[SERVER]', f'Salvato risultato: {col1} x {col2} = {result:.6f}')
                return result
            except Exception as e:
                logger.warning(f"Errore richiesta al client per {col1}-{col2}: {e}")
                print_status('[SERVER]', f'ERRORE chiamata client: {e}')
                return 0.0
        
        # Entrambe in chiaro: calcolo locale
        print_status('[SERVER]', f'Calcolo locale (dati in chiaro): {col1} x {col2}')
        # Entrambe numeriche
        if info1['is_numeric'] and info2['is_numeric']:
            return self._numeric_similarity(col1, col2)
        
        # Entrambe categoriche con unique_values disponibili
        elif info1['unique_values'] is not None and info2['unique_values'] is not None:
            return self._categorical_overlap(col1, col2)
        
        # Metrica basata su cardinalità
        else:
            return self._cardinality_similarity(col1, col2)
    
    def _numeric_similarity(self, col1: str, col2: str) -> float:
        """Calcola similarità tra colonne numeriche."""
        try:
            info1 = self.col_info[col1]
            info2 = self.col_info[col2]
            
            s1 = info1['numeric'] if info1['numeric'] is not None else info1['original']
            s2 = info2['numeric'] if info2['numeric'] is not None else info2['original']
            
            # Reset degli indici per garantire allineamento
            s1 = s1.reset_index(drop=True)
            s2 = s2.reset_index(drop=True)
            
            # Usa solo i primi N valori comuni
            min_len = min(len(s1), len(s2))
            if min_len < 2:
                return 0.0
            
            s1_aligned = s1.iloc[:min_len]
            s2_aligned = s2.iloc[:min_len]
            
            # Rimuovi NaN dopo l'allineamento
            mask = s1_aligned.notna() & s2_aligned.notna()
            s1_clean = s1_aligned[mask]
            s2_clean = s2_aligned[mask]
            
            if len(s1_clean) < 2:
                return 0.0
            
            # Correlazione di Pearson
            corr = s1_clean.corr(s2_clean)
            
            if pd.isna(corr):
                # Prova con Spearman come fallback
                from scipy.stats import spearmanr
                corr, _ = spearmanr(s1_clean, s2_clean)
            
            return abs(float(corr)) if not pd.isna(corr) else 0.0
            
        except Exception as e:
            logger.warning(f"Errore calcolo similarità numerica {col1}-{col2}: {e}")
            return 0.0
    
    def _categorical_overlap(self, col1: str, col2: str) -> float:
        """Calcola sovrapposizione tra colonne categoriche."""
        try:
            set1 = self.col_info[col1]['unique_values']
            set2 = self.col_info[col2]['unique_values']
            
            if not set1 or not set2:
                return 0.0
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return float(intersection) / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Errore calcolo overlap categorico {col1}-{col2}: {e}")
            return 0.0
    
    def _cardinality_similarity(self, col1: str, col2: str) -> float:
        """
        Metrica di similarità basata sulla cardinalità.
        Utile quando non si può calcolare correlazione o overlap.
        """
        try:
            n1 = self.col_info[col1]['nunique']
            n2 = self.col_info[col2]['nunique']
            
            if n1 == 0 or n2 == 0:
                return 0.0
            
            # Similarità basata sul rapporto di cardinalità
            # Valori simili di cardinalità → similarità più alta
            ratio = min(n1, n2) / max(n1, n2)
            
            return float(ratio * 0.5)  # Scala a 0-0.5 per distinguerla da altre metriche
            
        except Exception as e:
            logger.warning(f"Errore calcolo similarità cardinalità {col1}-{col2}: {e}")
            return 0.0

@app.route("/evaluate-column", methods=["POST"])
def evaluate_column():
    """
    Endpoint per valutare le colonne del dataset e generare matrice DC.
    
    Request body:
    {
        "path_df": "path/to/dataset.csv",
        "path": "path/to/output"
    }
    
    Returns:
        JSON con status e path ai file generati
    """
    try:
        # Validazione input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Nessun dato ricevuto"}), 400
        
        path_df = data.get('path_df')
        output_path = data.get('path')
        
        if not path_df:
            return jsonify({"error": "path_df mancante"}), 400
        if not output_path:
            return jsonify({"error": "path mancante"}), 400
        
        print_status('[SERVER]', f'File ricevuto dal client: {path_df}')
        print_status('[SERVER]', f'File esiste: {os.path.exists(path_df)}')
        logger.info(f"[SERVER] Inizio elaborazione: {path_df}")
        
        # 1. Carica e converti dataset
        h5_path, df = DataFrameLoader.load_and_convert(path_df)
        print_status('[SERVER]', f'Dataset caricato: shape={df.shape}, columns={list(df.columns[:3])}...')
        logger.info(f"Dataset caricato: {df.shape}")
        
        # 2. Recupera threshold globali
        global json_item
        thresholds = json_item if json_item else {}
        
        # 3. Costruisci matrice DC
        builder = DCMatrixBuilder(df, thresholds=thresholds)
        dc_matrix, rank_mapping = builder.build_predicate_matrix()
        
        # 3. Salva risultati
        os.makedirs(output_path, exist_ok=True)
        
        matrix_path = os.path.join(output_path, 'Matrix_DCs.csv')
        mapping_path = os.path.join(output_path, 'Rank_map.json')
        
        dc_matrix.to_csv(matrix_path, index=True)  # index=True per includere nomi colonne
        salva_json(dati=rank_mapping, percorso_file=mapping_path)
        
        logger.info("[SERVER] Elaborazione completata")
        
        return jsonify({
            "status": "SUCCESS",
            "message": "Matrice DC generata con successo",
            "output": {
                "matrix": matrix_path,
                "mapping": mapping_path,
                "h5_path": h5_path
            },
            "stats": {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "matrix_shape": list(dc_matrix.shape)
            }
        }), 200
        
    except FileNotFoundError as e:
        logger.error(f"File non trovato: {e}")
        return jsonify({"error": f"File non trovato: {str(e)}"}), 404
    
    except ValueError as e:
        logger.error(f"Errore di validazione: {e}")
        return jsonify({"error": f"Valore non valido: {str(e)}"}), 400
    
    except Exception as e:
        logger.error(f"Errore interno: {e}", exc_info=True)
        return jsonify({"error": f"Errore interno: {str(e)}"}), 500
# Avvio del server Flask
app.run(port=5000, threaded=True, use_reloader=False)
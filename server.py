"""   Faiss con @ e numba.jit   definire parametri di gestione , usare un lm matrix
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
import json
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

json_item = {}  #CONTA LE SOGHLIE DI TRASHOLD RICEVUTE DAL CLIENT
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
        # Prima prova a leggere un oggetto DataFrame salvato come '/df'
        with h5py.File(path, 'r') as h5file:
            if 'df' in h5file:

                print_status('[SERVER]', f'DEBUG: Trovato /df, caricamento con pd.read_hdf')
                return pd.read_hdf(path, key='/df')

        # Se non esiste '/df', ricostruisci il DataFrame leggendo tutte
        # le chiavi disponibili nello store (supporta sia 'col_<name>'
        # che chiavi plain come '<name>'). Questo evita di restituire
        # solo la prima colonna quando il file contiene dataset separati.
        with pd.HDFStore(path, mode='r') as store:
            keys = [k.strip('/') for k in store.keys()]

            # Se trovi chiavi con prefisso 'col_' usa la funzione esistente
            col_keys = [k for k in keys if k.startswith('col_')]
            if col_keys:
                columns = [k.replace('col_', '') for k in col_keys]
                return load_columns_from_h5(path, columns)

            # Altrimenti concatena tutti i dataset presenti nello store
            if keys:
                df_parts = []
                for raw_key in store.keys():
                    part = store[raw_key]
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
                
                return df_full

            raise ValueError("Nessuna chiave valida trovata nel file HDF5")


<<<<<<< Updated upstream
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
        
        for col in self.columns:
            series = self.df[col].dropna()
            
            # Controlla se la colonna è marcata come sensibile nei threshold
            col_threshold = self.thresholds.get(col, {})
            is_sensitive = col_threshold.get('sensitive', False)
            
            if is_sensitive:
                # Colonna cifrata (sensibile): memorizza info minime
                #print_status('[SERVER]', f'Colonna CIFRATA rilevata: {col} (da threshold)')
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
=======
class DCPredicateGenerator:
    """
    Genera e valuta predicati atomici per Denial Constraints.
    """
    
    # Operatori supportati
    OPERATORS = {
        'EQ': '==',   # Uguale
        'NEQ': '<>',  # Diverso
        'LT': '<',    # Minore
        'LEQ': '<=',  # Minore o uguale
        'GT': '>',    # Maggiore
        'GEQ': '>='   # Maggiore o uguale
    }
    
    def __init__(self, df: pd.DataFrame, thresholds: Dict = None, use_column_indices: bool = True):
        """
        Args:
            df: DataFrame del dataset (usato SOLO per metadati: nomi colonne e n_rows)
            thresholds: Dizionario con info su colonne sensibili
            use_column_indices: Se True usa COL0, COL1... invece dei nomi reali
            
        SICUREZZA:
        Il server NON memorizza i dati del dataframe.
        Memorizza solo: nomi colonne, numero righe.
        Tutte le operazioni sui dati vengono delegate al client.
        """
        # SOLO METADATI - NON memorizziamo self.df!
        self.original_columns = df.columns.tolist()
        self.n_rows = len(df)  # Solo il conteggio, non i dati
        self.n_cols = len(self.original_columns)
        self.thresholds = thresholds or {}
        self.use_column_indices = use_column_indices
        
        # Mappa nome_originale -> COL{index}
        self.col_to_index = {col: f"COL{i}" for i, col in enumerate(self.original_columns)}
        self.index_to_col = {v: k for k, v in self.col_to_index.items()}
        
        # NOTA: Non identifichiamo più colonne "sensibili" vs "non sensibili"
        # TUTTE le operazioni passano dal client per sicurezza
        
        logger.info(f"DCPredicateGenerator: {self.n_cols} colonne, {self.n_rows} righe (server NON vede dati)")
    
    def generate_predicates(self, max_per_pair: int = 6) -> List[Dict]:
        """
        Genera tutti i predicati atomici da testare.
        
        Args:
            max_per_pair: Numero massimo di predicati per coppia di colonne
            
        Returns:
            Lista di dizionari con info predicati
        """
        predicates = []
        
        # Genera predicati per ogni coppia di colonne
        for i in range(self.n_cols):
            for j in range(self.n_cols):
                col1_orig = self.original_columns[i]
                col2_orig = self.original_columns[j]
                
                col1 = self.col_to_index[col1_orig] if self.use_column_indices else col1_orig
                col2 = self.col_to_index[col2_orig] if self.use_column_indices else col2_orig
                
                # Genera predicati per tutti gli operatori
                for op_name, op_symbol in self.OPERATORS.items():
                    predicates.append({
                        'col1': col1,
                        'col2': col2,
                        'col1_orig': col1_orig,
                        'col2_orig': col2_orig,
                        'operator': op_name,
                        'operator_symbol': op_symbol,
                        'predicate_id': f"{col1}_{col2}_{op_name}"
                    })
        
        logger.info(f"Generati {len(predicates)} predicati atomici")
        return predicates
    
    def evaluate_predicates(self, predicates: List[Dict], sample_size: int = None) -> Dict:
        """
        Valuta tutti i predicati su coppie di righe del dataset.
        
        SICUREZZA:
        Il server NON accede MAI ai dati. Per OGNI predicato chiede al client
        di valutarlo e riceve SOLO il risultato (conteggio + coppie soddisfatte).
        
        Args:
            predicates: Lista predicati da valutare
            sample_size: Numero di coppie di righe da campionare (None = tutte)
            
        Returns:
            Dizionario con statistiche per ogni predicato
        """
        n_rows = self.n_rows  # Usa il conteggio memorizzato, non self.df
        
        # Genera coppie di righe da confrontare (solo indici, non dati!)
        if sample_size and sample_size < (n_rows * (n_rows - 1)) // 2:
            import random
            all_pairs = [(i, j) for i in range(n_rows) for j in range(i+1, n_rows)]
            row_pairs = random.sample(all_pairs, min(sample_size, len(all_pairs)))
            logger.info(f"Campionamento: {len(row_pairs)} coppie casuali")
        else:
            # Limita a max 500 righe per evitare esplosione combinatoria
            max_rows = min(500, n_rows)
            row_pairs = [(i, j) for i in range(max_rows) for j in range(i+1, max_rows)]
            logger.info(f"Valutazione completa: {len(row_pairs)} coppie (max {max_rows} righe)")
        
        total_pairs = len(row_pairs)
        results = {}
        
        logger.info(f"Inizio valutazione {len(predicates)} predicati su {total_pairs} coppie...")
        logger.info(f"[SICUREZZA] Server NON accede ai dati - TUTTE le valutazioni via client callback")
        
        # Valuta ogni predicato - SEMPRE via client, mai locale!
        for idx, pred in enumerate(predicates):
            if idx % 100 == 0:
                logger.info(f"Progresso: {idx}/{len(predicates)} predicati valutati")
            
            pred_id = pred['predicate_id']
            col1_orig = pred['col1_orig']
            col2_orig = pred['col2_orig']
            operator = pred['operator']
            
            satisfied = 0
            satisfied_pairs = []
            
            # SEMPRE chiedi al client - il server NON ha accesso ai dati!
            try:
                if idx < 5 or idx % 100 == 0:  # Stampa primi 5 e ogni 100
                    logger.info(f"[{idx}] Valutazione predicato: {col1_orig} {operator} {col2_orig}")
                
                result = request_operation_to_client(
                    operation="evaluate_predicate",
                    col1=col1_orig,
                    col2=col2_orig,
                    operator=operator,
                    row_pairs=row_pairs
                )
                # Il client ritorna un dict con count e pairs
                if isinstance(result, dict):
                    satisfied = result.get('count', 0)
                    satisfied_pairs = result.get('pairs', [])
                    if idx < 5:  # Stampa dettagli primi 5
                        logger.info(f"[{idx}] Ricevuto dict: satisfied={satisfied}/{total_pairs}, pairs={len(satisfied_pairs)}")
                        logger.info(f"[{idx}]   Dict keys: {list(result.keys())}, count value: {result.get('count')}")
                else:
                    # Retrocompatibilità: se ritorna solo numero
                    satisfied = int(result) if result else 0
                    satisfied_pairs = []
                    if idx < 5:
                        logger.warning(f"[{idx}] Formato vecchio: solo conteggio {satisfied}, senza pairs, type={type(result)}")
            except Exception as e:
                logger.error(f"[{idx}] ERRORE valutazione predicato {pred_id} via client: {e}")
                logger.error(f"[{idx}]   Traceback: {repr(e)}")
                satisfied = 0
                satisfied_pairs = []
            
            # Calcola statistiche
            support = satisfied / total_pairs if total_pairs > 0 else 0
            confidence = support  # Semplificato: confidence = support per predicati atomici
            
            results[pred_id] = {
                'predicate': pred,
                'satisfied': satisfied,
                'total': total_pairs,
                'support': support,
                'confidence': confidence,
                'satisfied_pairs': satisfied_pairs  # Coppie soddisfatte
            }
        
        logger.info(f"Valutazione completata: {len(results)} predicati valutati")
        return results
    
    # NOTA: _evaluate_predicate_local() è stato RIMOSSO per sicurezza.
    # Il server NON deve MAI accedere ai dati in chiaro.
    # Tutte le valutazioni passano tramite client callback.
    
    def save_results(self, results: Dict, output_path: str):
        """
        Salva i risultati in formato JSON.
        """
        # DC_Predicates.json - lista completa predicati
        predicates_file = os.path.join(output_path, 'DC_Predicates.json')
        predicates_list = [v['predicate'] for v in results.values()]
        
        with open(predicates_file, 'w', encoding='utf-8') as f:
            json.dump(predicates_list, f, indent=2)
        
        logger.info(f"Salvato: {predicates_file}")
        
        # DC_Predicate_Stats.json - statistiche
        stats_file = os.path.join(output_path, 'DC_Predicate_Stats.json')
        stats_dict = {
            k: {
                'support': v['support'],
                'confidence': v['confidence'],
                'satisfied': v['satisfied'],
                'total': v['total'],
                'col1': v['predicate']['col1'],
                'col2': v['predicate']['col2'],
                'operator': v['predicate']['operator_symbol'],
                'satisfied_pairs': v.get('satisfied_pairs', [])  # Aggiungi le coppie
            }
            for k, v in results.items()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2)
        
        logger.info(f"Salvato: {stats_file}")
        
        return predicates_file, stats_file


class DCMatrixBuilder:
    """
    Costruisce la matrice dei vincoli di denial (Denial Constraints).
    
    SICUREZZA:
    Questa classe NON accede MAI ai dati in chiaro.
    Memorizza solo metadati (nomi colonne, numero righe).
    Tutte le operazioni di calcolo (similarity, metriche) vengono
    delegate al client tramite callback.
    """
    
    def __init__(self, df: pd.DataFrame, thresholds: Dict = None):
        """
        Args:
            df: DataFrame usato SOLO per estrarre metadati (nomi colonne, n_rows)
            thresholds: Dizionario con info su colonne (non usato per decidere se cifrare)
            
        SICUREZZA: Non memorizziamo self.df!
        """
        # SOLO METADATI - NON memorizziamo i dati!
        self.columns = df.columns.tolist()
        self.n_cols = len(self.columns)
        self.n_rows = len(df)
        self.thresholds = thresholds or {}
        
        logger.info(f"DCMatrixBuilder: {self.n_cols} colonne, {self.n_rows} righe (server NON vede dati)")
    
    def build_predicate_matrix(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Costruisce la matrice di similarità tra colonne.
        
        SICUREZZA:
        Per OGNI coppia di colonne, chiede al client di calcolare la similarità.
        Il server riceve SOLO il valore numerico risultante, mai i dati.
        """
        logger.info("Costruzione matrice similarità (TUTTE le operazioni via client)...")
>>>>>>> Stashed changes
        
        matrix = np.zeros((self.n_cols, self.n_cols))
        rank_mapping = {}
        
<<<<<<< Updated upstream
        for i, col1 in enumerate(self.columns):
            for j in range(i, self.n_cols):
                col2 = self.columns[j]
                
                if i == j:
                    metric = self._compute_diagonal_metric(col1)
                else:
                    metric = self._compute_pairwise_metric(col1, col2)
=======
        total_pairs = (self.n_cols * (self.n_cols + 1)) // 2
        processed = 0
        
        for i, col1 in enumerate(self.columns):
            for j in range(i, self.n_cols):
                col2 = self.columns[j]
                processed += 1
                
                if processed % 50 == 0:
                    logger.info(f"Progresso matrice: {processed}/{total_pairs} coppie")
                
                if i == j:
                    # Diagonale: chiedi metrica al client
                    metric = self._request_diagonal_metric(col1)
                else:
                    # Fuori diagonale: chiedi similarità al client
                    metric = self._request_similarity(col1, col2)
>>>>>>> Stashed changes
                
                matrix[i, j] = metric
                matrix[j, i] = metric
                
                rank_mapping[f"{col1}_{col2}"] = {
                    "col1": col1,
                    "col2": col2,
<<<<<<< Updated upstream
                    "metric_value": float(metric),
=======
                    "metric_value": float(metric), #risultato salvato 
>>>>>>> Stashed changes
                    "rank": i * self.n_cols + j,
                    "col1_index": i,
                    "col2_index": j
                }
        
        df_matrix = pd.DataFrame(matrix, index=self.columns, columns=self.columns)
        
<<<<<<< Updated upstream
        logger.info(f"Matrice DC costruita: {df_matrix.shape}")
=======
        logger.info(f"Matrice similarità costruita: {df_matrix.shape}")
>>>>>>> Stashed changes
        logger.info(f"Valori non-zero fuori diagonale: {np.count_nonzero(matrix - np.diag(np.diag(matrix)))}")
        
        return df_matrix, rank_mapping
    
<<<<<<< Updated upstream
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
        """Calcola COSINE SIMILARITY tra due colonne diverse."""
        info1 = self.col_info[col1]
        info2 = self.col_info[col2]
        
        # Se ALMENO UNA colonna è cifrata, richiedi al client
        if info1.get('is_encrypted') or info2.get('is_encrypted'):
            try:
                # Richiedi al client di calcolare la similarità
                result = request_operation_to_client(
                    operation="similarity",
                    col1=col1,
                    col2=col2
                )
                return result
            except Exception as e:
                logger.warning(f"Errore richiesta al client per {col1}-{col2}: {e}")
                return 0.0
        
        # Entrambe in chiaro: calcolo cosine similarity locale
        return self._cosine_similarity(col1, col2)
    
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
    
    def _cosine_similarity(self, col1: str, col2: str) -> float:
        """
        Calcola COSINE SIMILARITY tra due colonne.
        
        SPIEGAZIONE MATEMATICA:
        La cosine similarity misura l'angolo tra due vettori nello spazio n-dimensionale.
        Formula: cos(θ) = (A·B) / (||A|| × ||B||)
        dove:
        - A·B = prodotto scalare (dot product) tra i vettori
        - ||A|| = norma (lunghezza) del vettore A
        - ||B|| = norma (lunghezza) del vettore B
        
        Risultato: valore tra -1 e 1 (0 = perpendicolari, 1 = identici, -1 = opposti)
        Noi usiamo il valore assoluto per avere un range 0-1.
        """
        try:
            # STEP 1: Recupera le informazioni delle due colonne da confrontare
            info1 = self.col_info[col1]
            info2 = self.col_info[col2]
            
            # STEP 2: Converti entrambe le colonne in vettori numerici
            # Questo è necessario perché la cosine similarity lavora con numeri.
            # Se la colonna contiene stringhe, verranno codificate numericamente.
            vec1 = self._convert_to_numeric_vector(info1['original'])
            vec2 = self._convert_to_numeric_vector(info2['original'])
            
            # STEP 3: Allinea le lunghezze dei vettori
            # Le colonne potrebbero avere lunghezze diverse, prendiamo il minimo comune
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]  # Taglia il primo vettore alla lunghezza minima
            vec2 = vec2[:min_len]  # Taglia il secondo vettore alla lunghezza minima
            
            # STEP 4: Rimuovi i valori NaN (Not a Number)
            # I NaN causerebbero errori nel calcolo, quindi creiamo una maschera
            # per mantenere solo i valori validi in entrambi i vettori
            mask = ~(np.isnan(vec1) | np.isnan(vec2))  # True dove entrambi sono validi
            vec1_clean = vec1[mask]  # Filtra il primo vettore
            vec2_clean = vec2[mask]  # Filtra il secondo vettore
            
            # STEP 5: Controllo di sicurezza - servono almeno 2 valori per calcolare la similarity
            if len(vec1_clean) < 2:
                return 0.0
            
            # STEP 6: Calcola la COSINE SIMILARITY vera e propria
            
            # 6a) Calcola il prodotto scalare (dot product): A·B = Σ(a_i × b_i)
            # Es: [1,2,3] · [4,5,6] = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
            dot_product = np.dot(vec1_clean, vec2_clean)
            
            # 6b) Calcola la norma (lunghezza) del primo vettore: ||A|| = √(Σa_i²)
            # Es: ||[1,2,3]|| = √(1² + 2² + 3²) = √(1+4+9) = √14 ≈ 3.74
            norm1 = np.linalg.norm(vec1_clean)
            
            # 6c) Calcola la norma del secondo vettore
            norm2 = np.linalg.norm(vec2_clean)
            
            # 6d) Controllo divisione per zero - se un vettore ha norma 0, non possiamo calcolare
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # 6e) Calcola la cosine similarity finale: cos(θ) = (A·B) / (||A|| × ||B||)
            # Questo valore rappresenta il coseno dell'angolo tra i due vettori
            cosine_sim = dot_product / (norm1 * norm2)
            
            # STEP 7: Ritorna il valore assoluto per avere sempre un range [0, 1]
            # 1.0 = massima similarità (vettori identici o opposti)
            # 0.0 = minima similarità (vettori perpendicolari, nessuna correlazione)
            return abs(float(cosine_sim))
            
        except Exception as e:
            logger.warning(f"Errore calcolo cosine similarity {col1}-{col2}: {e}")
            return 0.0
    
    def _convert_to_numeric_vector(self, series: pd.Series) -> np.ndarray:
        """
        Converte una Series in vettore numerico.
        - Numerica → usa direttamente
        - Categorica → LabelEncoder
        """
        from sklearn.preprocessing import LabelEncoder
        
        # Prova conversione numerica diretta
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # Se > 50% convertiti → numerico
        if numeric_series.notna().sum() > len(series) * 0.5:
            return numeric_series.fillna(0).values
        
        # Altrimenti: encoding categorico
        le = LabelEncoder()
        series_filled = series.fillna('__NAN__').astype(str)
        encoded = le.fit_transform(series_filled)
        
        return encoded.astype(float)
=======
    def _request_diagonal_metric(self, col: str) -> float:
        """
        Richiede al client la metrica diagonale per una colonna.
        (es: range per numeriche, cardinalità normalizzata per categoriche)
        """
        try:
            result = request_operation_to_client(
                operation="diagonal_metric",
                col1=col,
                col2=col  # Stessa colonna
            )
            return float(result) if result else 1.0
        except Exception as e:
            logger.warning(f"Errore metrica diagonale per {col}: {e}")
            return 1.0  # Default placeholder
    
    def _request_similarity(self, col1: str, col2: str) -> float:
        """
        Richiede al client di calcolare la cosine similarity tra due colonne.
        Il server riceve SOLO il valore numerico, mai i dati.
        """
        try:
            result = request_operation_to_client(
                operation="similarity",
                col1=col1,
                col2=col2
            )
            return float(result) if result else 0.0
        except Exception as e:
            logger.warning(f"Errore similarity {col1}-{col2}: {e}")
            return 0.0


# NOTA: I seguenti metodi sono stati RIMOSSI per sicurezza:
# - _preprocess_data() - accedeva ai dati
# - _compute_diagonal_metric() - accedeva ai dati  
# - _compute_pairwise_metric() - accedeva ai dati
# - _numeric_similarity() - accedeva ai dati
# - _categorical_overlap() - accedeva ai dati
# - _cardinality_similarity() - accedeva ai dati
# - _cosine_similarity() - accedeva ai dati
# - _convert_to_numeric_vector() - accedeva ai dati
#
# Tutte le operazioni sono ora delegate al client via callback.

>>>>>>> Stashed changes

@app.route("/evaluate-column", methods=["POST"])
def evaluate_column():
    """
<<<<<<< Updated upstream
    Endpoint per valutare le colonne del dataset e generare matrice DC.
=======
    Endpoint per valutare le colonne del dataset e generare matrice DC + predicati.
>>>>>>> Stashed changes
    
    Request body:
    {
        "path_df": "path/to/dataset.csv",
<<<<<<< Updated upstream
        "path": "path/to/output"
=======
        "path": "path/to/output",
        "generate_predicates": true,  // Opzionale: genera anche predicati DC
        "sample_size": 10000           // Opzionale: campiona N coppie di righe
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
        generate_predicates = data.get('generate_predicates', True)  # Default: True
        sample_size = data.get('sample_size', None)
>>>>>>> Stashed changes
        
        if not path_df:
            return jsonify({"error": "path_df mancante"}), 400
        if not output_path:
            return jsonify({"error": "path mancante"}), 400
        
        logger.info(f"[SERVER] Inizio elaborazione: {path_df}")
<<<<<<< Updated upstream
=======
        logger.info(f"[SERVER] Genera predicati: {generate_predicates}")
>>>>>>> Stashed changes
        
        # 1. Carica e converti dataset
        h5_path, df = DataFrameLoader.load_and_convert(path_df)
        logger.info(f"Dataset caricato: {df.shape}")
        
        # 2. Recupera threshold globali
        global json_item
        thresholds = json_item if json_item else {}
        
<<<<<<< Updated upstream
        # 3. Costruisci matrice DC
        builder = DCMatrixBuilder(df, thresholds=thresholds)
        dc_matrix, rank_mapping = builder.build_predicate_matrix()
        
        # 3. Salva risultati
        os.makedirs(output_path, exist_ok=True)
        
        matrix_path = os.path.join(output_path, 'Matrix_DCs.csv')
        mapping_path = os.path.join(output_path, 'Rank_map.json')
        
        dc_matrix.to_csv(matrix_path, index=True)  # index=True per includere nomi colonne
        salva_json(dati=rank_mapping, percorso_file=mapping_path)
        
=======
        # 3. Costruisci matrice DC (similarità)
        logger.info("[SERVER] Costruzione matrice similarità...")
        builder = DCMatrixBuilder(df, thresholds=thresholds)
        dc_matrix, rank_mapping = builder.build_predicate_matrix()
        
        # 4. Salva matrice similarità
        os.makedirs(output_path, exist_ok=True)
        
        matrix_path = os.path.join(output_path, 'Matrix_Similarity.csv')
        mapping_path = os.path.join(output_path, 'Rank_map.json')
        
        dc_matrix.to_csv(matrix_path, index=True)
        salva_json(dati=rank_mapping, percorso_file=mapping_path)
        
        logger.info(f"[SERVER] Matrice salvata: {matrix_path}")
        
        output_files = {
            "matrix": matrix_path,
            "mapping": mapping_path,
            "h5_path": h5_path
        }
        
        # 5. Genera e valuta predicati DC (opzionale)
        predicates_evaluated = 0
        if generate_predicates:
            logger.info("[SERVER] Generazione predicati DC...")
            pred_generator = DCPredicateGenerator(df, thresholds=thresholds, use_column_indices=True)
            
            predicates = pred_generator.generate_predicates()
            logger.info(f"[SERVER] Valutazione {len(predicates)} predicati...")
            
            results = pred_generator.evaluate_predicates(predicates, sample_size=sample_size)
            
            # Salva risultati
            pred_file, stats_file = pred_generator.save_results(results, output_path)
            
            output_files['predicates'] = pred_file
            output_files['predicate_stats'] = stats_file
            predicates_evaluated = len(predicates)
            
            logger.info("[SERVER] Predicati valutati e salvati")
        
>>>>>>> Stashed changes
        logger.info("[SERVER] Elaborazione completata")
        
        return jsonify({
            "status": "SUCCESS",
<<<<<<< Updated upstream
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
=======
            "message": "Matrice DC e predicati generati con successo",
            "output": output_files,
            "stats": {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "matrix_shape": list(dc_matrix.shape),
                "predicates_evaluated": predicates_evaluated
>>>>>>> Stashed changes
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
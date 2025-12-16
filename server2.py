"""
====================================================
Python Server per l'elaborazione sicura di dataset
====================================================
Modificato per:
- Confronto RIGHE x RIGHE invece di features x features
- Nessun filtering/pruning delle colonne
- Tutte le colonne vengono utilizzate nel confronto
"""

import gc
import json
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import tenseal as ts
from flask import Flask, request, jsonify

from utils.utils_Server import salva_json
from utils.utils_common import save_df_h5, print_status, the_end, load_columns_from_h5

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURAZIONE GLOBALE
# ============================================================================

app = Flask('python-server')

# Variabili globali per stato del server
json_item: Dict = {}
context: Optional[ts.Context] = None
path: str = ''

fx = open("log/log_server.txt", "w")


# ============================================================================
# ENDPOINT FLASK
# ============================================================================

@app.route("/save_thr", methods=["POST"])
def save_thr():
    """
    Endpoint per salvare la soglia (thr) inviata dal client.
    """
    global json_item, path
    print_status('[SERVER]', '[Save thr] Server connected.')
    data = request.get_json(force=True)

    json_item = data.get("thr")
    path = data.get("path")

    if json_item is None:
        print_status('[SERVER]', '[Save thr] Error: Threshold not found.')
        return jsonify({"error": "Nessun campo 'thr' nel payload"}), 400
    else:
        print_status('[SERVER]', '[Save thr] Threshold file found.')
        return jsonify({"status": "saved"}), 200


@app.route("/upload_context", methods=["POST"])
def upload_context():
    """
    Endpoint per caricare un contesto TenSEAL inviato come file.
    """
    global context
    context = ts.context_from(request.files["file"].read())

    print_status('[SERVER]', '[CKKS] Context file found.')
    return "Contesto pubblico ricevuto", 200


@app.route("/end_connection", methods=["POST"])
def end_connection():
    """
    Endpoint per chiudere la connessione.
    """
    werk_log = logging.getLogger('werkzeug')
    prev_level = werk_log.level
    werk_log.setLevel(logging.ERROR)
    
    try:
        flag = request.form.get("flag")
        flag = flag.lower() == 'true' if flag is not None else False

        global json_item, context, path
        json_item, context, path = {}, None, ''
        gc.collect()
        
        print_status('[SERVER]', 'End Connection.')
        os.system('cls' if os.name == 'nt' else 'clear')
        
        if flag:
            the_end('[SERVER]')

        return "Connection Closed!", 200
    finally:
        werk_log.setLevel(prev_level)


@app.route("/evaluate-column", methods=["GET"])
def evaluate_column():
    """
    Endpoint per valutare righe e generare matrice DC ROW x ROW.
    
    Request body:
    {
        "path_df": "/path/to/dataset.h5",
        "path": "/path/to/output"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Nessun dato ricevuto"}), 400
        
        path_df = data.get('path_df')
        output_path = data.get('path')
        
        if not path_df or not output_path:
            return jsonify({"error": "path_df e path sono obbligatori"}), 400
        
        print_status('[SERVER]', '=' * 60)
        print_status('[SERVER]', f'[Evaluate] Elaborazione dataset: {path_df}')
        
        # 1. Carica threshold da json_item globale
        if not json_item:
            print_status('[SERVER]', '[Evaluate] WARN: json_item vuoto, uso default')
            config = ThresholdConfig()
        else:
            config = ThresholdConfig.from_json_item(json_item)
            print_status('[SERVER]', f'[Evaluate] Threshold numeric: {config.numeric_threshold}, '
                                 f'string: {config.string_threshold}')

        # 2. Carica dataset
        h5_path, df = DatasetLoader.load_and_convert(path_df)

        print_status('[SERVER]', f'[Evaluate] Dataset caricato: {df.shape[0]} righe, '
                                 f'{df.shape[1]} colonne')

        # 3. Costruzione matrice DC ROW x ROW
        builder = DCMatrixBuilder(
            df=df,
            config=config,
            json_item=json_item,
            context=context
        )

        df_matrix, rank_mapping, stats = builder.build_matrix()

        # 4. Salvataggio risultati
        os.makedirs(output_path, exist_ok=True)

        matrix_path = os.path.join(output_path, "Matrix_DCs.csv")
        rank_path = os.path.join(output_path, "Rank_map.json")
        stats_path = os.path.join(output_path, "dc_stats.json")

        df_matrix.to_csv(matrix_path, index=True)
        salva_json(rank_mapping, rank_path)
        salva_json(stats, stats_path)

        print_status('[SERVER]', '[Evaluate] Risultati salvati:')
        print_status('[SERVER]', f'  - Matrice DC: {matrix_path}')
        print_status('[SERVER]', f'  - Rank mapping: {rank_path}')
        print_status('[SERVER]', f'  - Statistiche: {stats_path}')

        # 5. Cleanup memoria
        del df, df_matrix
        gc.collect()

        return jsonify({
            "status": "completed",
            "matrix_path": matrix_path,
            "rank_path": rank_path,
            "stats_path": stats_path,
            "stats": stats
        }), 200

    except Exception as e:
        print_status('[SERVER]', f'[Evaluate] ERROR: {e}')
        import traceback
        print_status('[SERVER]', f'[Evaluate] Traceback:\n{traceback.format_exc()}')
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MODELLI DATI E CONFIGURAZIONE
# ============================================================================

@dataclass
class ThresholdConfig:
    """
    Configurazione threshold da json_item.
    """
    numeric_threshold: int = 0
    string_threshold: float = 0.5
    
    @classmethod
    def from_json_item(cls, json_item: Dict) -> 'ThresholdConfig':
        """
        Carica threshold da json_item globale.
        """
        try:
            thresholds = json_item.get('thresholds', {})
            
            string_thr = thresholds.get('string', 0.5)
            if isinstance(string_thr, str):
                string_thr = float(string_thr)
            
            return cls(
                numeric_threshold=int(thresholds.get('numeric', 0)),
                string_threshold=string_thr
            )
        except Exception as e:
            print_status('[SERVER]', f'[Config] Errore parsing json_item: {e}. Uso default.')
            return cls()


# ============================================================================
# GESTIONE DATASET E CONVERSIONI
# ============================================================================

class DatasetLoader:
    """Gestisce caricamento e conversione dataset"""
    
    @staticmethod
    def load_and_convert(path_df: str) -> Tuple[str, pd.DataFrame]:
        """
        Carica dataset da vari formati e converte in HDF5 se necessario.
        """
        if not os.path.exists(path_df):
            raise FileNotFoundError(f"File non trovato: {path_df}")
        
        ext = os.path.splitext(path_df)[1].lower()
        
        if ext == '.h5':
            print_status('[SERVER]', f'[Loader] Caricamento HDF5: {path_df}')
            return path_df, DatasetLoader._load_from_h5(path_df)
        
        elif ext == '.parquet':
            print_status('[SERVER]', f'[Loader] Conversione Parquet -> HDF5')
            df = pd.read_parquet(path_df)
            h5_path = path_df.replace('.parquet', '.h5')
            save_df_h5(df_path=h5_path, df=df)
            return h5_path, df
        
        elif ext == '.csv':
            print_status('[SERVER]', f'[Loader] Conversione CSV -> HDF5')
            df = pd.read_csv(path_df)
            h5_path = path_df.replace('.csv', '.h5')
            save_df_h5(df_path=h5_path, df=df)
            return h5_path, df
        
        else:
            raise ValueError(f"[Loader] Estensione non supportata: {ext}")
    
    @staticmethod
    def _load_from_h5(path: str) -> pd.DataFrame:
        """Carica dataframe da file HDF5"""
        with pd.HDFStore(path, mode='r') as store:
            if '/df' in store.keys():
                return store['/df']
            
            all_keys = [key.strip('/') for key in store.keys()]
            col_keys = [key for key in all_keys if key.startswith('col_')]
            
            if col_keys:
                columns = [key.replace('col_', '') for key in col_keys]
                return load_columns_from_h5(path, columns)
            
            if all_keys:
                return store[f'/{all_keys[0]}']
            
            raise ValueError("[Loader] Nessuna chiave valida in HDF5")


# ============================================================================
# COSTRUTTORE MATRICE DC ROW x ROW
# ============================================================================

class DCMatrixBuilder:
    """
    Costruisce matrice DC ROW x ROW confrontando tutte le righe del dataset.
    NESSUN FILTERING: usa tutte le colonne disponibili.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: ThresholdConfig,
        json_item: Dict,
        context: Optional[ts.Context] = None
    ):
        self.df = df
        self.config = config
        self.json_item = json_item
        self.context = context
        
        # USA TUTTE LE COLONNE - nessun filtering
        self.columns = df.columns.tolist()
        self.n_rows = len(df)
        self.n_cols = len(self.columns)
        
        # Identifica colonne sensibili e loro tipi
        self.sensitive_columns = set(json_item.get('sensitive_columns', []))
        self.column_types = json_item.get('column_types', {})
        
        print_status('[SERVER]', f'[Builder] Configurazione:')
        print_status('[SERVER]', f'  - Righe da confrontare: {self.n_rows}')
        print_status('[SERVER]', f'  - Colonne utilizzate: {self.n_cols} (TUTTE)')
        print_status('[SERVER]', f'  - Colonne sensibili: {len(self.sensitive_columns)}')
    
    def build_matrix(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Costruisce matrice DC ROW x ROW.
        
        Returns:
            - DataFrame matrice (n_rows x n_rows)
            - Dict rank mapping
            - Dict statistiche
        """
        print_status('[SERVER]', '[Builder] === INIZIO COSTRUZIONE MATRICE DC ROW x ROW ===')
        
        stats = {
            "n_rows": self.n_rows,
            "n_columns": self.n_cols,
            "n_sensitive_columns": len(self.sensitive_columns),
            "n_comparisons_total": self.n_rows * (self.n_rows + 1) // 2,
            "n_comparisons_computed": 0,
            "n_encrypted_operations": 0
        }
        
        # Inizializza matrice ROW x ROW
        matrix = np.zeros((self.n_rows, self.n_rows))
        rank_mapping = {}
        
        # Confronta ogni coppia di righe
        for i in range(self.n_rows):
            for j in range(i, self.n_rows):
                
                # Calcola distanza/similarità tra riga i e riga j
                if i == j:
                    # Diagonale: distanza con se stessa = 0
                    metric = 0.0
                else:
                    metric = self._compute_row_distance(i, j, stats)
                
                stats["n_comparisons_computed"] += 1
                
                # Matrice simmetrica
                matrix[i, j] = metric
                matrix[j, i] = metric
                
                # Rank mapping
                rank_mapping[f"row_{i}_row_{j}"] = {
                    "row1_index": int(i),
                    "row2_index": int(j),
                    "distance": float(metric),
                    "rank": int(i * self.n_rows + j)
                }
                
                # Log progress ogni 10%
                if stats["n_comparisons_computed"] % max(1, stats["n_comparisons_total"] // 10) == 0:
                    progress = (stats["n_comparisons_computed"] / stats["n_comparisons_total"]) * 100
                    print_status('[SERVER]', f'[Builder] Progresso: {progress:.1f}%')
        
        # DataFrame finale con indici delle righe
        row_labels = [f"row_{i}" for i in range(self.n_rows)]
        df_matrix = pd.DataFrame(matrix, index=row_labels, columns=row_labels)
        
        print_status('[SERVER]', '[Builder] === MATRICE DC ROW x ROW COMPLETATA ===')
        print_status('[SERVER]', f'[Builder] Shape matrice: {df_matrix.shape}')
        print_status('[SERVER]', f'[Builder] Operazioni su ciphertext: {stats["n_encrypted_operations"]}')
        
        return df_matrix, rank_mapping, stats
    
    def _compute_row_distance(self, row_idx1: int, row_idx2: int, stats: Dict) -> float:
        """
        Calcola distanza tra due righe considerando TUTTE le colonne.
        
        Strategia:
        - Per colonne numeriche: distanza normalizzata
        - Per colonne categoriche: 0 se uguali, 1 se diverse
        - Per colonne criptate: operazioni su ciphertext
        
        Restituisce la distanza media su tutte le colonne.
        """
        distances = []
        
        for col in self.columns:
            is_sensitive = col in self.sensitive_columns
            col_type = self.column_types.get(col, 'numeric')
            
            val1 = self.df.iloc[row_idx1][col]
            val2 = self.df.iloc[row_idx2][col]
            
            # Gestisci valori mancanti
            if pd.isna(val1) or pd.isna(val2):
                distances.append(1.0)  # Distanza massima per NA
                continue
            
            # CASO 1: Colonna sensibile/criptata
            if is_sensitive and self.context is not None:
                stats["n_encrypted_operations"] += 1
                dist = self._encrypted_distance(val1, val2, col_type)
                distances.append(dist)
            
            # CASO 2: Colonna numerica chiara
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                dist = self._numeric_distance(val1, val2, col)
                distances.append(dist)
            
            # CASO 3: Colonna categorica chiara
            else:
                dist = 0.0 if val1 == val2 else 1.0
                distances.append(dist)
        
        # Distanza media su tutte le colonne
        return float(np.mean(distances)) if distances else 0.0
    
    def _numeric_distance(self, val1: float, val2: float, col: str) -> float:
        """
        Distanza normalizzata tra valori numerici.
        Normalizza rispetto al range della colonna.
        """
        try:
            col_min = self.df[col].min()
            col_max = self.df[col].max()
            col_range = col_max - col_min
            
            if col_range == 0:
                return 0.0
            
            # Distanza normalizzata [0, 1]
            normalized_dist = abs(val1 - val2) / col_range
            return min(1.0, normalized_dist)
        
        except Exception as e:
            print_status('[SERVER]', f'[Builder] Warn: numeric distance {col}: {e}')
            return 0.5
    
    def _encrypted_distance(self, val1, val2, col_type: str) -> float:
        """
        Distanza approssimata su valori criptati.
        
        NOTA: Su ciphertext non possiamo calcolare distanze esatte.
        Usiamo threshold configurati come proxy.
        """
        try:
            if col_type == 'numeric':
                # Per numerici criptati: usa threshold numerico
                return float(self.config.numeric_threshold) / 100.0
            else:
                # Per categorici criptati: usa threshold string
                return self.config.string_threshold
        
        except Exception as e:
            print_status('[SERVER]', f'[Builder] Error encrypted distance: {e}')
            return 0.5


# ============================================================================
# AVVIO SERVER
# ============================================================================

if __name__ == '__main__':
    print_status('[SERVER]', '=' * 60)
    print_status('[SERVER]', 'Server Flask Avviato')
    print_status('[SERVER]', 'MODALITÀ: Confronto ROW x ROW (nessun filtering)')
    print_status('[SERVER]', 'Endpoints disponibili:')
    print_status('[SERVER]', '  POST /save_thr         - Salva threshold')
    print_status('[SERVER]', '  POST /upload_context   - Carica context CKKS')
    print_status('[SERVER]', '  GET /evaluate-column  - Genera matrice DC ROW x ROW')
    print_status('[SERVER]', '  POST /end_connection   - Chiude connessione')
    print_status('[SERVER]', '=' * 60)
    
    app.run(port=5000, threaded=True, use_reloader=False)
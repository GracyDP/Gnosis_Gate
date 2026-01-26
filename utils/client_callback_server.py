"""
====================================================
Client Callback Server
====================================================
Mini-server Flask che gira sul client per ricevere richieste
di operazioni su dati cifrati dal server principale.

Il server principale chiama questo endpoint quando ha bisogno
di operazioni su dati cifrati che solo il client può decriptare.
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from utils.utils_common import print_status
import tenseal as ts

app = Flask('client-callback')

# Variabili globali per contesto e dati
client_context = None
encrypted_data = {}

def set_client_context(context):
    """Imposta il contesto di decriptazione"""
    global client_context
    client_context = context

def set_encrypted_data(data_dict):
    """Imposta i dati cifrati disponibili"""
    global encrypted_data
    encrypted_data = data_dict

@app.route("/decrypt_and_compute", methods=["POST"])
def decrypt_and_compute():
    """
    Riceve richiesta di operazione dal server, usa i dati ORIGINALI in chiaro,
    esegue l'operazione e restituisce SOLO IL RISULTATO.
    
    IMPORTANTE: Non decripta nulla - i dati originali sono già in chiaro.
    Il server chiama questa funzione per operazioni su colonne che LUI vede cifrate,
    ma il client ha i dati originali in chiaro.
    """
    global client_context, encrypted_data
    
    try:
        data = request.get_json()
        operation = data.get('operation')
        col1 = data.get('col1')
        col2 = data.get('col2')
        
        print_status('[CLIENT-CALLBACK]', f'📥 RICHIESTA dal server: {operation}({col1}, {col2})')
        
        # Recupera il DataFrame originale
        original_df = encrypted_data.get('__original_df__')
        if original_df is None:
            error_msg = "DataFrame originale non configurato"
            print_status('[CLIENT-CALLBACK]', f'ERRORE: {error_msg}')
            return jsonify({"error": error_msg}), 400
        
        print_status('[CLIENT-CALLBACK]', f'Colonne disponibili: {list(original_df.columns[:5])}...')
        
        # Verifica che le colonne esistano
        if col1 not in original_df.columns:
            error_msg = f"Colonna {col1} non trovata nel dataset"
            print_status('[CLIENT-CALLBACK]', f'ERRORE: {error_msg}')
            return jsonify({"error": error_msg}), 400
            
        if col2 not in original_df.columns:
            error_msg = f"Colonna {col2} non trovata nel dataset"
            print_status('[CLIENT-CALLBACK]', f'ERRORE: {error_msg}')
            return jsonify({"error": error_msg}), 400
        
        # Estrai i dati in chiaro
        print_status('[CLIENT-CALLBACK]', f'Estrazione dati in chiaro...')
        data1 = original_df[col1].values
        data2 = original_df[col2].values
        
        print_status('[CLIENT-CALLBACK]', f'   Col1: {len(data1)} valori, tipo={type(data1[0]).__name__}')
        print_status('[CLIENT-CALLBACK]', f'   Col2: {len(data2)} valori, tipo={type(data2[0]).__name__}')
        
        # Esegui l'operazione richiesta
        print_status('[CLIENT-CALLBACK]', f'Esecuzione operazione: {operation}')
        
        # Usa sempre "similarity" che replica la logica del server
        if operation in ["correlation", "similarity"]:
            result = _compute_similarity_like_server(data1, data2)
        elif operation == "dot_product":
            result = _compute_dot_product(data1, data2)
        elif operation == "distance":
            result = _compute_distance(data1, data2)
        else:
            error_msg = f"Operazione non supportata: {operation}"
            print_status('[CLIENT-CALLBACK]', f'ERRORE: {error_msg}')
            return jsonify({"error": error_msg}), 400
        
        print_status('[CLIENT-CALLBACK]', f'RISULTATO: {result:.6f} - Invio al server')
        
        return jsonify({
            "result": float(result),
            "operation": operation,
            "cols": [col1, col2]
        }), 200
        
    except Exception as e:
        print_status('[CLIENT-CALLBACK]', f'ERRORE: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _compute_similarity_like_server(data1, data2):
    """
    Replica ESATTAMENTE la logica del server per calcolare similarità.
    Stesso codice di DCMatrixBuilder._compute_pairwise_metric per dati in chiaro.
    """
    import pandas as pd
    
    # Converti in Series pandas
    s1 = pd.Series(data1)
    s2 = pd.Series(data2)
    
    # Tenta conversione numerica
    numeric1 = pd.to_numeric(s1, errors='coerce')
    numeric2 = pd.to_numeric(s2, errors='coerce')
    
    is_numeric1 = numeric1.notna().sum() > len(s1) * 0.5
    is_numeric2 = numeric2.notna().sum() > len(s2) * 0.5
    
    # CASO 1: Entrambe numeriche → Correlazione di Pearson
    if is_numeric1 and is_numeric2:
        print_status('[CLIENT-CALLBACK]', f'   Entrambe numeriche - correlazione')
        return _numeric_similarity(numeric1, numeric2)
    
    # CASO 2: Entrambe categoriche con pochi valori unici → Jaccard overlap
    unique1 = set(s1.unique()) if len(s1) < 10000 else None
    unique2 = set(s2.unique()) if len(s2) < 10000 else None
    
    if unique1 is not None and unique2 is not None:
        print_status('[CLIENT-CALLBACK]', f'   📑 Entrambe categoriche → overlap')
        return _categorical_overlap(unique1, unique2)
    
    # CASO 3: Fallback → Similarità basata su cardinalità
    print_status('[CLIENT-CALLBACK]', f'   Fallback - cardinalità')
    return _cardinality_similarity(s1.nunique(), s2.nunique())


def _numeric_similarity(s1, s2):
    """Calcola similarità tra colonne numeriche (STESSO CODICE DEL SERVER)"""
    try:
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
        print_status('[CLIENT-CALLBACK]', f'   Errore correlazione: {e}')
        return 0.0


def _categorical_overlap(set1, set2):
    """Calcola sovrapposizione tra colonne categoriche (STESSO CODICE DEL SERVER)"""
    try:
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return float(intersection) / union if union > 0 else 0.0
        
    except Exception as e:
        print_status('[CLIENT-CALLBACK]', f'   Errore overlap: {e}')
        return 0.0


def _cardinality_similarity(n1, n2):
    """Similarità basata su cardinalità (STESSO CODICE DEL SERVER)"""
    try:
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Similarità basata sul rapporto di cardinalità
        ratio = min(n1, n2) / max(n1, n2)
        
        return float(ratio * 0.5)  # Scala a 0-0.5 per distinguerla da altre metriche
        
    except Exception as e:
        print_status('[CLIENT-CALLBACK]', f'   Errore cardinalità: {e}')
        return 0.0


def _compute_correlation(data1, data2):
    """Calcola correlazione tra due array"""
    arr1 = np.array(data1)
    arr2 = np.array(data2)
    
    # Verifica tipi
    is_numeric1 = np.issubdtype(arr1.dtype, np.number)
    is_numeric2 = np.issubdtype(arr2.dtype, np.number)
    
    # Se almeno uno non è numerico, prova a convertire
    if not is_numeric1:
        arr1 = pd.to_numeric(pd.Series(arr1), errors='coerce').values
    if not is_numeric2:
        arr2 = pd.to_numeric(pd.Series(arr2), errors='coerce').values
    
    # Allineamento lunghezze
    min_len = min(len(arr1), len(arr2))
    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]
    
    # Rimuovi NaN
    mask = ~(np.isnan(arr1) | np.isnan(arr2))
    arr1 = arr1[mask]
    arr2 = arr2[mask]
    
    if len(arr1) < 2:
        print_status('[CLIENT-CALLBACK]', f'   Troppi pochi valori validi ({len(arr1)}), ritorno 0.0')
        return 0.0
    
    # Correlazione di Pearson
    corr = np.corrcoef(arr1, arr2)[0, 1]
    return abs(corr) if not np.isnan(corr) else 0.0


def _compute_dot_product(data1, data2):
    """Calcola prodotto scalare"""
    arr1 = np.array(data1)
    arr2 = np.array(data2)
    min_len = min(len(arr1), len(arr2))
    return float(np.dot(arr1[:min_len], arr2[:min_len]))


def _compute_distance(data1, data2):
    """Calcola distanza euclidea"""
    arr1 = np.array(data1)
    arr2 = np.array(data2)
    min_len = min(len(arr1), len(arr2))
    return float(np.linalg.norm(arr1[:min_len] - arr2[:min_len]))


def _compute_similarity(data1, data2):
    """Calcola similarità (1 - distanza normalizzata)"""
    distance = _compute_distance(data1, data2)
    # Normalizza la distanza
    max_dist = max(np.linalg.norm(data1), np.linalg.norm(data2))
    if max_dist == 0:
        return 1.0
    return 1.0 - min(distance / max_dist, 1.0)


def start_callback_server(port=5001):
    """Avvia il server di callback in un thread separato"""
    print_status('[CLIENT-CALLBACK]', f'Avvio server callback su porta {port}')
    app.run(port=port, threaded=True, use_reloader=False)

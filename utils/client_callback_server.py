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
<<<<<<< Updated upstream
=======
    
    OPERAZIONI SUPPORTATE:
    - similarity: calcola cosine similarity tra due colonne
    - diagonal_metric: calcola statistiche della diagonale (una sola colonna)
    - evaluate_predicate: valuta predicato su coppie di righe
    - correlation, dot_product, distance: altre metriche
>>>>>>> Stashed changes
    """
    global client_context, encrypted_data
    
    try:
        data = request.get_json()
        operation = data.get('operation')
        col1 = data.get('col1')
<<<<<<< Updated upstream
        col2 = data.get('col2')
=======
        col2 = data.get('col2')  # Può essere None per diagonal_metric
>>>>>>> Stashed changes
        
        # Recupera il DataFrame originale
        original_df = encrypted_data.get('__original_df__')
        if original_df is None:
            error_msg = "DataFrame originale non configurato"
            print_status('[CLIENT-CALLBACK]', f' ERRORE: {error_msg}')
            return jsonify({"error": error_msg}), 400
        
<<<<<<< Updated upstream
        # Verifica che le colonne esistano
=======
        # ========================================
        # OPERAZIONE: DIAGONAL_METRIC
        # Calcola statistiche per una singola colonna (diagonale della matrice)
        # ========================================
        if operation == "diagonal_metric":
            if col1 not in original_df.columns:
                return jsonify({"error": f"Colonna {col1} non trovata"}), 400
            
            col_data = original_df[col1]
            
            # Calcola statistiche
            nunique = col_data.nunique()
            n_rows = len(col_data)
            is_numeric = pd.api.types.is_numeric_dtype(col_data)
            
            # Metrica diagonale: quanto è "distintiva" la colonna
            # Alta cardinalità → valore alto (colonna più informativa)
            diagonal_value = nunique / n_rows if n_rows > 0 else 0.0
            
            print_status('[CLIENT-CALLBACK]', f'Diagonal metric {col1}: {diagonal_value:.4f} (unique={nunique}, rows={n_rows})')
            
            return jsonify({
                "result": float(diagonal_value),
                "operation": operation,
                "col": col1,
                "stats": {
                    "nunique": int(nunique),
                    "n_rows": int(n_rows),
                    "is_numeric": bool(is_numeric)
                }
            }), 200
        
        # Per le altre operazioni servono entrambe le colonne
>>>>>>> Stashed changes
        if col1 not in original_df.columns:
            error_msg = f"Colonna {col1} non trovata nel dataset"
            print_status('[CLIENT-CALLBACK]', f'ERRORE: {error_msg}')
            return jsonify({"error": error_msg}), 400
            
        if col2 not in original_df.columns:
            error_msg = f"Colonna {col2} non trovata nel dataset"
            print_status('[CLIENT-CALLBACK]', f'ERRORE: {error_msg}')
            return jsonify({"error": error_msg}), 400
        
        # Estrai i dati in chiaro
        data1 = original_df[col1].values
        data2 = original_df[col2].values
        
        # Esegui l'operazione richiesta
        
        # Usa sempre "similarity" che replica la logica del server
        if operation in ["correlation", "similarity"]:
            result = _compute_similarity_like_server(data1, data2)
        elif operation == "dot_product":
            result = _compute_dot_product(data1, data2)
        elif operation == "distance":
            result = _compute_distance(data1, data2)
<<<<<<< Updated upstream
=======
        elif operation == "evaluate_predicate":
            # Valuta predicato su coppie di righe
            operator = data.get('operator')
            row_pairs = data.get('row_pairs', [])
            
            if not operator or not row_pairs:
                return jsonify({"error": "Mancano 'operator' o 'row_pairs'"}), 400
            
            print_status('[CLIENT-CALLBACK]', f'Valuto predicato: {col1} {operator} {col2} su {len(row_pairs)} coppie')
            
            # Ritorna sia conteggio che coppie soddisfatte
            count, pairs = _evaluate_predicate_on_pairs_with_list(data1, data2, operator, row_pairs)
            result = {'count': count, 'pairs': pairs}
            
            print_status('[CLIENT-CALLBACK]', f'Risultato: {count}/{len(row_pairs)} soddisfatte, {len(pairs)} pairs salvate')
            print_status('[CLIENT-CALLBACK]', f'  DEBUG: result type={type(result)}, content={result}')
>>>>>>> Stashed changes
        else:
            error_msg = f"Operazione non supportata: {operation}"
            print_status('[CLIENT-CALLBACK]', f'ERRORE: {error_msg}')
            return jsonify({"error": error_msg}), 400
        
<<<<<<< Updated upstream
        return jsonify({
            "result": float(result),
            "operation": operation,
            "cols": [col1, col2]
        }), 200
=======
        # IMPORTANTE: Se result è già un dict (es: evaluate_predicate), NON convertire in float!
        if isinstance(result, dict):
            return jsonify({
                "result": result,
                "operation": operation,
                "cols": [col1, col2]
            }), 200
        else:
            return jsonify({
                "result": float(result),
                "operation": operation,
                "cols": [col1, col2]
            }), 200
>>>>>>> Stashed changes
        
    except Exception as e:
        print_status('[CLIENT-CALLBACK]', f'ERRORE: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/evaluate_predicates", methods=["POST"])
def evaluate_predicates():
    """
    Valuta predicati riga per riga per la costruzione di Denial Constraints.
    
    Il server invia:
    - Lista di predicati da valutare (es: [{"col1": "COL0", "col2": "COL1", "op": "NEQ"}])
    - Opzionalmente: coppie di righe specifiche da confrontare
    
    Il client risponde con:
    - Statistiche di supporto per ogni predicato (quante coppie di righe lo soddisfano)
    """
    global encrypted_data
    
    try:
        data = request.get_json()
        predicates = data.get('predicates', [])  # Lista di predicati da valutare
        row_pairs = data.get('row_pairs', None)  # Coppie di righe specifiche (opzionale)
        sample_size = data.get('sample_size', None)  # Numero di campioni (opzionale)
        
        print_status('[CLIENT-CALLBACK]', f'Valutazione {len(predicates)} predicati...')
        
        # Recupera il DataFrame originale
        original_df = encrypted_data.get('__original_df__')
        if original_df is None:
            return jsonify({"error": "DataFrame originale non configurato"}), 400
        
        n_rows = len(original_df)
        print_status('[CLIENT-CALLBACK]', f' Dataset: {n_rows} righe')
        
        # Se non specificate, genera coppie di righe da confrontare
        if row_pairs is None:
            if sample_size and sample_size < (n_rows * (n_rows - 1)) // 2:
                # Campionamento casuale per dataset grandi
                import random
                all_pairs = [(i, j) for i in range(n_rows) for j in range(i+1, n_rows)]
                row_pairs = random.sample(all_pairs, min(sample_size, len(all_pairs)))
                print_status('[CLIENT-CALLBACK]', f' Campionamento: {len(row_pairs)} coppie casuali')
            else:
                # Tutte le coppie di righe distinte
                max_rows = min(1000, n_rows)  # Limita per evitare esplosione combinatoria
                row_pairs = [(i, j) for i in range(max_rows) for j in range(i+1, max_rows)]
                print_status('[CLIENT-CALLBACK]', f'Valutazione completa: {len(row_pairs)} coppie')
        
        # Valuta ogni predicato su tutte le coppie di righe
        results = []
        
        for pred_idx, predicate in enumerate(predicates):
            col1_name = predicate['col1']
            col2_name = predicate['col2']
            operator = predicate['op']  # 'EQ', 'NEQ', 'LT', 'LE', 'GT', 'GE'
            
            # Verifica che le colonne esistano
            if col1_name not in original_df.columns or col2_name not in original_df.columns:
                print_status('[CLIENT-CALLBACK]', f'WARNING: Colonne {col1_name}/{col2_name} non trovate')
                continue
            
            col1_data = original_df[col1_name].values
            col2_data = original_df[col2_name].values
            
            # Valuta il predicato per ogni coppia di righe
            satisfied_count = 0
            satisfied_pairs = []
            
            for row_i, row_j in row_pairs:
                val1 = col1_data[row_i]  # t0.col1
                val2 = col2_data[row_j]  # t1.col2
                
                is_satisfied = _evaluate_predicate(val1, val2, operator)
                
                if is_satisfied:
                    satisfied_count += 1
                    if len(satisfied_pairs) < 100:  # Salva solo i primi 100 esempi
                        satisfied_pairs.append([int(row_i), int(row_j)])
            
            support = satisfied_count / len(row_pairs) if row_pairs else 0
            
            results.append({
                'predicate_index': pred_idx,
                'col1': col1_name,
                'col2': col2_name,
                'operator': operator,
                'satisfied_count': satisfied_count,
                'total_pairs': len(row_pairs),
                'support': float(support),
                'satisfied_pairs': satisfied_pairs  # Primi 100 esempi
            })
            
            if (pred_idx + 1) % 50 == 0:
                print_status('[CLIENT-CALLBACK]', f'Progresso: {pred_idx + 1}/{len(predicates)} predicati')
        
        print_status('[CLIENT-CALLBACK]', f'Valutazione completata: {len(results)} predicati')
        
        return jsonify({
            "predicates_evaluated": len(predicates),
            "row_pairs_evaluated": len(row_pairs),
            "results": results
        }), 200
        
    except Exception as e:
        print_status('[CLIENT-CALLBACK]', f'ERRORE valutazione predicati: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _evaluate_predicate(val1, val2, operator):
    """
    Valuta un singolo predicato tra due valori.
    
    Args:
        val1: valore dalla riga i, colonna 1 (t0.col1)
        val2: valore dalla riga j, colonna 2 (t1.col2)
<<<<<<< Updated upstream
        operator: 'EQ', 'NEQ', 'LT', 'LE', 'GT', 'GE'
=======
        operator: 'EQ', 'NEQ', 'LT', 'LEQ', 'GT', 'GEQ'
>>>>>>> Stashed changes
    
    Returns:
        bool: True se il predicato è soddisfatto
    """
    import pandas as pd
    
    # Gestisci NaN
    if pd.isna(val1) or pd.isna(val2):
        return False
    
    try:
        if operator == 'EQ':
            return val1 == val2
        elif operator == 'NEQ':
            return val1 != val2
        elif operator == 'LT':
<<<<<<< Updated upstream
            return val1 < val2
        elif operator == 'LE':
            return val1 <= val2
        elif operator == 'GT':
            return val1 > val2
        elif operator == 'GE':
            return val1 >= val2
=======
            return float(val1) < float(val2)
        elif operator in ['LEQ', 'LE']:
            return float(val1) <= float(val2)
        elif operator == 'GT':
            return float(val1) > float(val2)
        elif operator in ['GEQ', 'GE']:
            return float(val1) >= float(val2)
>>>>>>> Stashed changes
        else:
            return False
    except (TypeError, ValueError):
        # Se i valori non sono comparabili, considera solo EQ/NEQ
        if operator == 'EQ':
            return str(val1) == str(val2)
        elif operator == 'NEQ':
            return str(val1) != str(val2)
        return False


<<<<<<< Updated upstream
=======
def _evaluate_predicate_on_pairs(data1, data2, operator, row_pairs):
    """
    Valuta un predicato su coppie di righe e ritorna il conteggio.
    
    Args:
        data1: array di valori colonna 1
        data2: array di valori colonna 2
        operator: operatore da testare
        row_pairs: lista di tuple (i, j) - indici righe da confrontare
        
    Returns:
        int: numero di coppie che soddisfano il predicato
    """
    satisfied = 0
    
    for i, j in row_pairs:
        if i >= len(data1) or j >= len(data2):
            continue
        
        val1 = data1[i]
        val2 = data2[j]
        
        if _evaluate_predicate(val1, val2, operator):
            satisfied += 1
    
    return satisfied


def _evaluate_predicate_on_pairs_with_list(data1, data2, operator, row_pairs):
    """
    Valuta un predicato su coppie di righe e ritorna conteggio e lista coppie.
    
    Args:
        data1: array di valori colonna 1
        data2: array di valori colonna 2
        operator: operatore da testare
        row_pairs: lista di tuple (i, j) - indici righe da confrontare
        
    Returns:
        Tuple[int, List]: (numero coppie soddisfatte, lista coppie soddisfatte)
    """
    satisfied = 0
    satisfied_pairs = []
    debug_first = True  # Stampa debug prima coppia
    
    for i, j in row_pairs:
        if i >= len(data1) or j >= len(data2):
            continue
        
        val1 = data1[i]
        val2 = data2[j]
        
        is_satisfied = _evaluate_predicate(val1, val2, operator)
        
        # Debug: stampa prima coppia
        if debug_first:
            print_status('[CLIENT-CALLBACK]', f'  Prima coppia: [{i},{j}] val1={val1}, val2={val2}, op={operator} => {is_satisfied}')
            debug_first = False
        
        if is_satisfied:
            satisfied += 1
            # Salva solo primi 100 per non esplodere memoria
            if len(satisfied_pairs) < 100:
                satisfied_pairs.append([int(i), int(j)])
    
    return satisfied, satisfied_pairs


>>>>>>> Stashed changes
def _compute_similarity_like_server(data1, data2):
    """
    Calcola COSINE SIMILARITY vera tra due colonne (lato CLIENT).
    
    COSA FA QUESTA FUNZIONE:
    Quando il client deve calcolare la similarity tra due colonne,
    usa la stessa logica matematica del server per garantire coerenza.
    
    Formula matematica: cosine_similarity = (A · B) / (||A|| × ||B||)
    Dove:
    - A, B = i due vettori da confrontare
    - A · B = prodotto scalare (dot product)
    - ||A||, ||B|| = norme (lunghezze) dei vettori
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    # PREPARAZIONE: Converti i dati in formato pandas Series per manipolazione più facile
    s1 = pd.Series(data1)
    s2 = pd.Series(data2)
    
    # STEP 1: Converti entrambe le colonne in vettori numerici
    # Questo passaggio è cruciale perché la cosine similarity richiede numeri.
    # Se le colonne contengono testo (es: "rosso", "blu"), verranno codificate in numeri.
    vec1 = _convert_to_numeric_vector(s1)
    vec2 = _convert_to_numeric_vector(s2)
    
    # STEP 2: Allinea le lunghezze dei due vettori
    # Se una colonna ha 100 righe e l'altra 80, prendiamo solo le prime 80 di entrambe
    min_len = min(len(vec1), len(vec2))
    vec1 = vec1[:min_len]  # Tronca al minimo comune
    vec2 = vec2[:min_len]
    
    # STEP 3: Rimuovi le righe con valori NaN (mancanti) in uno o entrambi i vettori
    # Creiamo una "maschera" che indica quali posizioni hanno valori validi
    mask = ~(np.isnan(vec1) | np.isnan(vec2))  # True = entrambi validi, False = almeno uno è NaN
    vec1_clean = vec1[mask]  # Mantieni solo i valori validi del primo vettore
    vec2_clean = vec2[mask]  # Mantieni solo i valori validi del secondo vettore
    
    # Controllo di sicurezza: servono almeno 2 valori per calcolare una similarity significativa
    if len(vec1_clean) < 2:
        return 0.0
    
    # STEP 4: Calcola la COSINE SIMILARITY
    
    # 4a) PRODOTTO SCALARE (dot product): somma dei prodotti elemento per elemento
    # Esempio: [1,2,3] · [4,5,6] = (1*4) + (2*5) + (3*6) = 32
    dot_product = np.dot(vec1_clean, vec2_clean)
    
    # 4b) NORMA del primo vettore: lunghezza geometrica del vettore
    # Formula: ||v|| = sqrt(v1² + v2² + ... + vn²)
    # Esempio: ||[3,4]|| = sqrt(3² + 4²) = sqrt(9 + 16) = 5
    norm1 = np.linalg.norm(vec1_clean)
    
    # 4c) NORMA del secondo vettore
    norm2 = np.linalg.norm(vec2_clean)
    
    # 4d) Controllo divisione per zero: se un vettore è tutto zeri, non possiamo calcolare
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 4e) CALCOLO FINALE: dividi il prodotto scalare per il prodotto delle norme
    # Questo normalizza il risultato nell'intervallo [-1, 1]
    # Interpretazione:
    #   1.0 = vettori identici (stesso "orientamento")
    #   0.0 = vettori perpendicolari (nessuna correlazione)
    #  -1.0 = vettori opposti (correlazione negativa)
    cosine_sim = dot_product / (norm1 * norm2)
    
    # STEP 5: Ritorna il valore assoluto per avere sempre un range positivo [0, 1]
    # Questo perché nel nostro contesto ci interessa solo "quanto sono simili",
    # non se hanno correlazione positiva o negativa
    return abs(float(cosine_sim))


def _convert_to_numeric_vector(series):
    """
    Converte una Series pandas in vettore numerico.
    - Se già numerica: usa direttamente
    - Se categorica/stringa: usa LabelEncoder per codificarla
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    # Prova conversione numerica diretta
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Se > 50% valori convertiti con successo → usa come numerico
    if numeric_series.notna().sum() > len(series) * 0.5:
        return numeric_series.fillna(0).values
    
    # Altrimenti: codifica categorica con LabelEncoder
    le = LabelEncoder()
    # Riempi NaN con stringa placeholder prima dell'encoding
    series_filled = series.fillna('__NAN__').astype(str)
    encoded = le.fit_transform(series_filled)
    
    return encoded.astype(float)


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

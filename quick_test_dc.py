"""
Quick Test - Generazione Denial Constraints
===========================================

Script di test rapido per verificare la pipeline DC.
"""

import requests
import json
import os
import time

# Configurazione
SERVER_URL = "http://localhost:5000"
DATASET_PATH = "DB/hepatitis/hepatitis_preProcessed.csv"
OUTPUT_PATH = "Test/hepatitis/quick_test_output"

def test_pipeline():
    """Test completo della pipeline DC"""
    
    print("="*70)
    print("TEST RAPIDO - GENERAZIONE DENIAL CONSTRAINTS")
    print("="*70)
    
    # 1. Verifica server
    print("\n[1/5] Verifica connessione server...")
    try:
        resp = requests.get(SERVER_URL, timeout=2)
        print("     Server raggiungibile")
    except:
        print("     Server NON raggiungibile")
        print("     Avvia con: python server.py")
        return False
    
    # 2. Verifica dataset
    print("\n[2/5] Verifica dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"     Dataset non trovato: {DATASET_PATH}")
        return False
    
    import pandas as pd
    df = pd.read_csv(DATASET_PATH)
    print(f"     Dataset trovato: {df.shape[0]} righe, {df.shape[1]} colonne")
    
    # 3. Invio richiesta
    print("\n[3/5] Invio richiesta al server...")
    print("     (Questo può richiedere alcuni secondi...)")
    
    payload = {
        "path_df": DATASET_PATH,
        "path": OUTPUT_PATH,
        "generate_predicates": True,
        "sample_size": 500  # Limita a 500 coppie per test veloce
    }
    
    start_time = time.time()
    
    try:
        resp = requests.post(
            f"{SERVER_URL}/evaluate-column",
            json=payload,
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        if resp.status_code == 200:
            print(f"     Richiesta completata in {elapsed:.1f}s")
            result = resp.json()
        else:
            print(f"     Errore: {resp.status_code}")
            print(f"     {resp.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("     Timeout (>120s)")
        return False
    except Exception as e:
        print(f"     Errore: {e}")
        return False
    
    # 4. Verifica file generati
    print("\n[4/5] Verifica file generati...")
    
    expected_files = {
        'Matrix_Similarity.csv': 'Matrice similarità',
        'Rank_map.json': 'Ranking colonne',
        'DC_Predicates.json': 'Predicati generati',
        'DC_Predicate_Stats.json': 'Statistiche predicati'
    }
    
    all_ok = True
    for filename, desc in expected_files.items():
        filepath = os.path.join(OUTPUT_PATH, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"     {desc}: {size:,} bytes")
        else:
            print(f"     {desc}: NON TROVATO")
            all_ok = False
    
    if not all_ok:
        return False
    
    # 5. Statistiche
    print("\n[5/5] Statistiche risultati...")
    
    stats = result.get('stats', {})
    print(f"     • Righe dataset: {stats.get('n_rows')}")
    print(f"     • Colonne dataset: {stats.get('n_columns')}")
    print(f"     • Predicati valutati: {stats.get('predicates_evaluated')}")
    
    # Analizza DC_Predicate_Stats.json
    stats_file = os.path.join(OUTPUT_PATH, 'DC_Predicate_Stats.json')
    with open(stats_file, 'r') as f:
        pred_stats = json.load(f)
    
    # Conta predicati per support
    high_support = sum(1 for v in pred_stats.values() if v['support'] >= 0.8)
    med_support = sum(1 for v in pred_stats.values() if 0.5 <= v['support'] < 0.8)
    low_support = sum(1 for v in pred_stats.values() if v['support'] < 0.5)
    
    print(f"\n     Distribuzione Support:")
    print(f"        • Support >= 80%: {high_support} predicati")
    print(f"        • Support 50-80%: {med_support} predicati")
    print(f"        • Support < 50%: {low_support} predicati")
    
    # 6. Estrai DC
    print("\n[6/6] Estrazione Denial Constraints...")
    print("     Esegui manualmente:")
    print(f"\n     python extract_dcs.py \\")
    print(f"         {stats_file} \\")
    print(f"         {OUTPUT_PATH}/DCs_extracted.txt \\")
    print(f"         0.8")
    
    print("\n" + "="*70)
    print("TEST COMPLETATO CON SUCCESSO!")
    print("="*70)
    print(f"\nOutput in: {OUTPUT_PATH}")
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    exit(0 if success else 1)

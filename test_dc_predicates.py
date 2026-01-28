"""
Test Script per Valutazione Predicati DC
==========================================

Questo script testa il nuovo sistema di valutazione predicati
per Denial Constraints.

Esegui dopo aver avviato server e client.
"""

import requests
import json
import os

# Configurazione
SERVER_URL = "http://localhost:5000"
TEST_DATASET = "Test/hepatitis/All_Sensitive/hepatitis_Encrypted.h5"  # Modifica con il tuo path
OUTPUT_PATH = "Test/hepatitis/test_predicates_output"

def test_predicate_evaluation():
    """
    Test completo del sistema di valutazione predicati.
    """
    print("=" * 70)
    print("TEST VALUTAZIONE PREDICATI PER DENIAL CONSTRAINTS")
    print("=" * 70)
    
    # 1. Verifica che il server sia attivo
    print("\n1 Verifica connessione server...")
    try:
        response = requests.get(f"{SERVER_URL}/")
        print(f"   [OK] Server attivo: {response.status_code}")
    except Exception as e:
        print(f"   [ERROR] Server non raggiungibile: {e}")
        return
    
    # 2. Verifica che il dataset esista
    print("\n2 Verifica dataset...")
    if not os.path.exists(TEST_DATASET):
        print(f"   [WARNING] Dataset non trovato: {TEST_DATASET}")
        print(f"   [INFO] Modifica TEST_DATASET nel script con un path valido")
        return
    print(f"   [OK] Dataset trovato: {TEST_DATASET}")
    
    # 3. Invia richiesta di evaluation
    print("\n3. Invio richiesta valutazione...")
    payload = {
        "path_df": TEST_DATASET,
        "output_path": OUTPUT_PATH
    }
    
    try:
        response = requests.post(
            f"{SERVER_URL}/evaluate-column",
            json=payload,
            timeout=600  # 10 minuti
        )
        
        if response.status_code == 200:
            print(f"   [OK] Richiesta completata con successo!")
            result = response.json()
            
            # 4. Mostra risultati
            print("\n4. Risultati:")
            print(f"   Status: {result.get('status')}")
            print(f"   📝 Message: {result.get('message')}")
            
            if 'stats' in result:
                stats = result['stats']
                print(f"\n   📈 Statistiche Dataset:")
                print(f"      - Righe: {stats.get('n_rows')}")
                print(f"      - Colonne: {stats.get('n_columns')}")
                print(f"      - Predicati valutati: {stats.get('predicates_evaluated', 'N/A')}")
            
            if 'output' in result:
                output = result['output']
                print(f"\n   File generati:")
                for key, path in output.items():
                    if os.path.exists(path):
                        size = os.path.getsize(path)
                        print(f"      [OK] {key}: {path} ({size:,} bytes)")
                    else:
                        print(f"      [WARNING] {key}: {path} (non trovato)")
            
            # 5. Analizza i file JSON generati
            print("\n5. Analisi file predicati:")
            analyze_predicate_files(OUTPUT_PATH)
            
        else:
            print(f"   [ERROR] Errore: {response.status_code}")
            print(f"   📝 Risposta: {response.text[:500]}")
            
    except requests.exceptions.Timeout:
        print(f"   [TIMEOUT] Timeout - operazione troppo lunga (>10 min)")
    except Exception as e:
        print(f"   [ERROR] Errore: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(" Test completato!")
    print("=" * 70)


def analyze_predicate_files(output_path):
    """
    Analizza i file di predicati generati.
    """
    predicates_file = os.path.join(output_path, 'DC_Predicates.json')
    stats_file = os.path.join(output_path, 'DC_Predicate_Stats.json')
    
    # Analizza predicati dettagliati
    if os.path.exists(predicates_file):
        with open(predicates_file, 'r') as f:
            predicates = json.load(f)
        
        print(f"\n   DC_Predicates.json:")
        print(f"      - Predicati totali: {len(predicates)}")
        
        # Raggruppa per operatore
        ops_count = {}
        high_support = []
        
        for pred_key, pred_data in predicates.items():
            op = pred_data.get('operator', 'UNKNOWN')
            ops_count[op] = ops_count.get(op, 0) + 1
            
            support = pred_data.get('support', 0)
            if support > 0.8:  # Support > 80%
                high_support.append({
                    'key': pred_key,
                    'support': support,
                    'col1': pred_data.get('col1'),
                    'col2': pred_data.get('col2'),
                    'op': op
                })
        
        print(f"      - Operatori:")
        for op, count in sorted(ops_count.items()):
            print(f"         • {op}: {count}")
        
        print(f"\n      - Predicati con support > 80%: {len(high_support)}")
        if high_support:
            print(f"         Top 5:")
            for pred in sorted(high_support, key=lambda x: x['support'], reverse=True)[:5]:
                print(f"         • {pred['col1']} {pred['op']} {pred['col2']}: {pred['support']:.2%}")
    else:
        print(f"     DC_Predicates.json non trovato")
    
    # Analizza statistiche
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"\n    DC_Predicate_Stats.json:")
        print(f"      - Predicati: {len(stats)}")
        
        supports = [s.get('support', 0) for s in stats.values()]
        if supports:
            avg_support = sum(supports) / len(supports)
            print(f"      - Support medio: {avg_support:.2%}")
            print(f"      - Support min: {min(supports):.2%}")
            print(f"      - Support max: {max(supports):.2%}")
    else:
        print(f"   ⚠️  DC_Predicate_Stats.json non trovato")


def test_client_callback():
    """
    Test diretto del client callback endpoint.
    """
    print("\n" + "=" * 70)
    print("🧪 TEST DIRETTO CLIENT CALLBACK")
    print("=" * 70)
    
    CLIENT_URL = "http://localhost:5001"
    
    # Test con predicati semplici
    test_predicates = [
        {"col1": "COL0", "col2": "COL0", "op": "EQ"},
        {"col1": "COL0", "col2": "COL0", "op": "NEQ"},
        {"col1": "COL0", "col2": "COL1", "op": "LT"},
    ]
    
    payload = {
        "predicates": test_predicates,
        "sample_size": 100
    }
    
    try:
        print(f"\n📤 Invio {len(test_predicates)} predicati di test al client...")
        response = requests.post(
            f"{CLIENT_URL}/evaluate_predicates",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Risposta ricevuta:")
            print(f"   - Predicati valutati: {result['predicates_evaluated']}")
            print(f"   - Coppie valutate: {result['row_pairs_evaluated']}")
            
            if 'results' in result:
                print(f"\n   Risultati:")
                for res in result['results'][:3]:  # Primi 3
                    print(f"      • {res['col1']} {res['operator']} {res['col2']}")
                    print(f"        Support: {res['support']:.2%} ({res['satisfied_count']}/{res['total_pairs']})")
        else:
            print(f"❌ Errore: {response.status_code}")
            print(f"   Risposta: {response.text}")
            
    except Exception as e:
        print(f"❌ Errore: {e}")
        print(f"💡 Assicurati che il client sia avviato e il callback attivo")


if __name__ == "__main__":
    import sys
    
    print("\n" + "🚀 " * 20)
    print("SCRIPT DI TEST - DENIAL CONSTRAINTS PREDICATES")
    print("🚀 " * 20 + "\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "client":
            test_client_callback()
        elif sys.argv[1] == "full":
            test_predicate_evaluation()
        else:
            print("Uso: python test_dc_predicates.py [client|full]")
    else:
        # Default: test completo
        test_predicate_evaluation()

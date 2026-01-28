"""
Script per convertire DC_Predicate_Stats.json in formato CSV per FastADC
========================================================================

FastADC necessita di un formato specifico per leggere i predicati.
Questo script converte il JSON generato dal tuo sistema Python
in un formato leggibile da FastADC.

Uso:
    python create_input_for_fastadc.py [input_json] [output_csv]
    
Esempio:
    python create_input_for_fastadc.py \
        Test/hepatitis/All_Sensitive/output_N2_S0.75/DC_Predicate_Stats.json \
        Test/hepatitis/All_Sensitive/output_N2_S0.75/predicates_input.csv
"""

import json
import pandas as pd
import sys
import os
from pathlib import Path


def convert_json_to_fastadc_csv(input_json_path, output_csv_path):
    """
    Converte DC_Predicate_Stats.json in CSV per FastADC.
    
    Formato INPUT (JSON):
    {
      "COL0_COL1_NEQ": {
        "support": 0.85,
        "confidence": 0.85,
        "satisfied": 850,
        "total": 1000
      }
    }
    
    Formato OUTPUT (CSV):
    predicate_id,col1,col2,operator,support,confidence,satisfied,total
    0,COL0,COL1,NEQ,0.85,0.85,850,1000
    """
    
    print(f" Lettura JSON: {input_json_path}")
    
    # Leggi JSON
    with open(input_json_path, 'r') as f:
        stats = json.load(f)
    
    print(f"correttamente Caricati {len(stats)} predicati")
    
    # Converti in formato FastADC
    data = []
    
    for pred_idx, (pred_key, pred_stat) in enumerate(stats.items()):
        # Parse predicate key: "COL0_COL1_NEQ"
        parts = pred_key.split('_')
        
        # Gestisci colonne con underscore nel nome (es: "COL_0_COL_1_NEQ")
        # Trova l'operatore (ultimo elemento)
        operators = ['EQ', 'NEQ', 'LT', 'LE', 'GT', 'GE']
        operator = None
        
        for op in operators:
            if pred_key.endswith(f"_{op}"):
                operator = op
                # Rimuovi operatore dalla chiave
                cols_part = pred_key[:-len(f"_{op}")]
                # Dividi in due colonne (assume formato COL1_COL2)
                col_parts = cols_part.split('_')
                
                # Se numero pari, dividi a metà
                mid = len(col_parts) // 2
                col1 = '_'.join(col_parts[:mid])
                col2 = '_'.join(col_parts[mid:])
                break
        
        if operator is None:
            print(f"⚠️  Predicato senza operatore valido: {pred_key}")
            continue
        
        data.append({
            'predicate_id': pred_idx,
            'col1': col1,
            'col2': col2,
            'operator': operator,
            'support': pred_stat['support'],
            'confidence': pred_stat['confidence'],
            'satisfied': pred_stat['satisfied'],
            'total': pred_stat['total']
        })
    
    # Crea DataFrame
    df = pd.DataFrame(data)
    
    # Salva CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f" Salvato CSV: {output_csv_path}")
    print(f" Statistiche:")
    print(f"   - Predicati totali: {len(df)}")
    print(f"   - Operatori:")
    for op, count in df['operator'].value_counts().items():
        print(f"      • {op}: {count}")
    print(f"   - Support medio: {df['support'].mean():.2%}")
    print(f"   - Predicati con support > 80%: {(df['support'] > 0.8).sum()}")
    
    return df


def convert_json_to_fastadc_format(input_json_path, output_dir):
    """
    Converte JSON in formato diretto per FastADC (alternativa).
    
    Genera file nella struttura che FastADC si aspetta:
    - predicates.txt: lista predicati
    - support.txt: supporto per ogni predicato
    """
    
    print(f" Conversione formato FastADC nativo...")
    
    with open(input_json_path, 'r') as f:
        stats = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    predicates_file = os.path.join(output_dir, 'predicates.txt')
    support_file = os.path.join(output_dir, 'support.txt')
    
    with open(predicates_file, 'w') as f_pred, open(support_file, 'w') as f_supp:
        for pred_key, pred_stat in stats.items():
            # Converti formato: "COL0_COL1_NEQ" → "t0.COL0 <> t1.COL1"
            parts = pred_key.split('_')
            operators_map = {
                'EQ': '==',
                'NEQ': '<>',
                'LT': '<',
                'LE': '<=',
                'GT': '>',
                'GE': '>='
            }
            
            # Trova operatore
            operator = None
            for op_key, op_symbol in operators_map.items():
                if pred_key.endswith(f"_{op_key}"):
                    operator = op_symbol
                    cols_part = pred_key[:-len(f"_{op_key}")]
                    col_parts = cols_part.split('_')
                    mid = len(col_parts) // 2
                    col1 = '_'.join(col_parts[:mid])
                    col2 = '_'.join(col_parts[mid:])
                    break
            
            if operator:
                # Formato FastADC: t0.COL0 <> t1.COL1
                predicate_str = f"t0.{col1} {operator} t1.{col2}"
                f_pred.write(f"{predicate_str}\n")
                f_supp.write(f"{pred_stat['support']}\n")
    
    print(f" Generati file FastADC:")
    print(f"   - {predicates_file}")
    print(f"   - {support_file}")


def main():
    """Main function con argomenti da command line."""
    
    if len(sys.argv) < 2:
        print(" Uso: python create_input_for_fastadc.py <input_json> [output_csv]")
        print("\nEsempio:")
        print("  python create_input_for_fastadc.py DC_Predicate_Stats.json predicates.csv")
        print("\nOppure usa path di default:")
        print("  python create_input_for_fastadc.py")
        
        # Usa path di default
        use_default = input("\n Usare path di default? (y/n): ").strip().lower()
        if use_default != 'y':
            sys.exit(1)
        
        input_json = "Test/hepatitis/All_Sensitive/output_N2_S0.75/DC_Predicate_Stats.json"
        output_csv = "Test/hepatitis/All_Sensitive/output_N2_S0.75/predicates_input.csv"
    else:
        input_json = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else input_json.replace('.json', '.csv')
    
    # Verifica che il file esista
    if not os.path.exists(input_json):
        print(f" File non trovato: {input_json}")
        sys.exit(1)
    
    print("=" * 70)
    print(" CONVERSIONE JSON → CSV PER FASTADC")
    print("=" * 70)
    
    # Converti
    df = convert_json_to_fastadc_csv(input_json, output_csv)
    
    # Mostra preview
    print("\n Preview (prime 5 righe):")
    print(df.head().to_string())
    
    # Opzionale: genera anche formato nativo FastADC
    output_dir = os.path.dirname(output_csv)
    fastadc_dir = os.path.join(output_dir, 'fastadc_format')
    
    generate_native = input("\n Generare anche formato nativo FastADC? (y/n): ").strip().lower()
    if generate_native == 'y':
        convert_json_to_fastadc_format(input_json, fastadc_dir)
    
    print("\n" + "=" * 70)
    print(" CONVERSIONE COMPLETATA!")
    print("=" * 70)
    print(f"\n File generato: {output_csv}")
    print(f" Dimensione: {os.path.getsize(output_csv):,} bytes")
    print("\n Prossimi passi:")
    print(f"   1. cd FastADC")
    print(f"   2. java -jar target/FastADC-1.0.jar \\")
    print(f"         --input ../{output_csv} \\")
    print(f"         --output ../DCs_output.txt")


if __name__ == "__main__":
    main()

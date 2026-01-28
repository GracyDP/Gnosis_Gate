"""
====================================================
Utilità Server - Request to Client
====================================================
Funzioni per richiedere operazioni al client quando
il server ha bisogno di operare su dati cifrati.
"""

import requests
from utils.utils_common import print_status

CLIENT_CALLBACK_URL = "http://localhost:5001"  # URL del callback server sul client


def request_operation_to_client(
    operation: str,
    col1: str,
    col2: str,
    encrypted_data1=None,
    encrypted_data2=None
):
    """
    Richiede al client di eseguire un'operazione su dati cifrati.

    Returns:
        float: Risultato dell'operazione
    """

    payload = {
        "operation": operation,
        "col1": col1,
        "col2": col2
    }

    if encrypted_data1 is not None:
        payload["encrypted_data1"] = encrypted_data1
    if encrypted_data2 is not None:
        payload["encrypted_data2"] = encrypted_data2

    url = f"{CLIENT_CALLBACK_URL}/decrypt_and_compute"

    try:
        # 1. Invio richiesta
        response = requests.post(url, json=payload, timeout=30)
        
        # 2. Se c'è errore, mostra il contenuto della risposta
        if response.status_code != 200:
            print_status('[SERVER]', f'Codice errore: {response.status_code}')
            print_status('[SERVER]', f'Risposta: {response.text[:500]}')  # Primi 500 char
        
        response.raise_for_status()

        # 3. Parsing JSON
        try:
            result_data = response.json()
        except ValueError as e:
            raise ValueError(
                f"Risposta non JSON dal client: {response.text}"
            ) from e

        # 3. Estrazione risultato
        if "result" not in result_data:
            raise KeyError(
                f"Chiave 'result' mancante nella risposta: {result_data}"
            )

        result = result_data["result"]

        if result is None:
            raise ValueError("Risultato nullo ricevuto dal client")

        # 4. Conversione sicura a float
        try:
            result = float(result)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Risultato non convertibile in float: {result} ({type(result)})"
            ) from e

        return result

    except Exception as e:
        # Catch totale: niente errori silenziosi
        print_status(
            '[SERVER]',
            f'ERRORE nella richiesta al client: {repr(e)}'
        )
        raise


def request_predicate_evaluation_from_client(
    predicates: list,
    row_pairs=None,
    sample_size=None
):
    """
    Richiede al client di valutare una lista di predicati su coppie di righe.
    
    Questo è il cuore della discovery delle Denial Constraints:
    - Il server genera predicati (es: t0.COL0 <> t1.COL1)
    - Il client valuta riga per riga quali sono soddisfatti
    - Il server riceve solo statistiche aggregate (non i dati!)
    
    Args:
        predicates: Lista di dict con formato:
            [{"col1": "COL0", "col2": "COL1", "op": "NEQ"}, ...]
        row_pairs: Lista opzionale di coppie (i,j) da valutare
        sample_size: Numero di campioni casuali (se row_pairs è None)
    
    Returns:
        dict: {
            "predicates_evaluated": int,
            "row_pairs_evaluated": int,
            "results": [
                {
                    "col1": "COL0",
                    "col2": "COL1",
                    "operator": "NEQ",
                    "support": 0.85,  # 85% coppie soddisfano il predicato
                    "satisfied_count": 850,
                    "total_pairs": 1000
                },
                ...
            ]
        }
    """
    payload = {
        "predicates": predicates,
        "row_pairs": row_pairs,
        "sample_size": sample_size
    }
    
    url = f"{CLIENT_CALLBACK_URL}/evaluate_predicates"
    
    try:
        print_status('[SERVER]', f'Richiesta valutazione {len(predicates)} predicati al client...')
        
        response = requests.post(url, json=payload, timeout=300)  # 5 minuti per dataset grandi
        
        if response.status_code != 200:
            print_status('[SERVER]', f'Errore valutazione predicati: {response.status_code}')
            print_status('[SERVER]', f'Risposta: {response.text[:500]}')
        
        response.raise_for_status()
        result_data = response.json()
        
        print_status('[SERVER]', f'Ricevuti risultati per {result_data["predicates_evaluated"]} predicati')
        
        return result_data
        
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Timeout nella valutazione predicati (>5min)")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Errore connessione durante valutazione predicati: {e}")
    except Exception as e:
        print_status('[SERVER]', f'Errore richiesta predicati al client: {str(e)}')
        raise


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

    print_status(
        '[SERVER]',
        f'📤 RICHIESTA al client: {operation}({col1}, {col2})'
    )

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

        print_status('[SERVER]', f'📦 JSON ricevuto: {result_data}')

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

        # 5. Log finale
        print_status('[SERVER]', f'📨 RISPOSTA dal client: {result:.6f}')

        return result

    except Exception as e:
        # Catch totale: niente errori silenziosi
        print_status(
            '[SERVER]',
            f'ERRORE nella richiesta al client: {repr(e)}'
        )
        raise

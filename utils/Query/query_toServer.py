from flask import Flask
import requests
from utils.Homomorphic.homomorphic_client import decode_decrypt_ckks, ricevi_ckks
from utils.url.uri import load_uri
from utils.utils_common import print_status

app = Flask('python-client')

server_url = load_uri('server')

def delete_file(flag):
    url = f"{server_url.rstrip('/')}/end_connection"
    try:
        response = requests.post(url, data={'flag': flag})
        response.raise_for_status()
        print_status('[CLIENT] ',response.text)
        return True
    except requests.HTTPError as e:
        raise Exception(f"❌ HTTP error: {e}")
    except requests.RequestException as e:
        raise Exception(f"❌ Errore di connessione: {e}")
    except ValueError:
        raise Exception("❌ Risposta non JSON dal server")

# Alla fine del tuo file:
def main():
    app.run(port=5001, threaded=True, use_reloader=False)

if __name__ == '__main__':
    main()


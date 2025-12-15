import json
import os

def load_uri(uri):

    # Path assoluto relativo alla posizione di questo file
    json_path = os.path.join(os.path.dirname(__file__), 'url.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    if uri in 'client':
        return data['CLIENT_URL']
    elif uri in 'server':
        return data['SERVER_URL']

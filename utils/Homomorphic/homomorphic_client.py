import base64
import os
import shutil
import warnings

import requests
import tenseal as ts

from utils.url.uri import load_uri
from utils.utils_common import identifica_tipo

warnings.filterwarnings("ignore")

base_ckks = "utils/Homomorphic/ckks/"
ctx_path_sk_ckks = f'{base_ckks}ckks_secret.context'
ctx_path_pk_ckks = f'{base_ckks}ckks_public.context'

ctx_path_sk_ckks_test = f'ckks_secret.context'
ctx_path_pk_ckks_test = f'ckks_public.context'

base_bfv = 'utils/Homomorphic/bfv/'
ctx_path_sk_bfv = f'{base_bfv}bfv_secret.context'
ctx_path_pk_bfv = f'{base_bfv}bfv_public.context'

ctx_path_sk_bfv_test = f'bfv_secret.context'
ctx_path_pk_bfv_test = f'bfv_public.context'

server_url = load_uri('server')

def create_context(schema, path, create_ctx=True):
    base_path = f'{path}/Homomorphic{schema}/'
    os.makedirs(base_path, exist_ok=True)
    if create_ctx:

        if schema == 'CKKS':
            try:
                context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=16384,
                    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
                )
                context.global_scale = 2 ** 40
                context.generate_galois_keys()

                secret_bytes_sk = context.serialize(save_secret_key=True,
                                                    save_public_key=True,
                                                    save_relin_keys=True,
                                                    save_galois_keys=True)

                # ctx_path_sck = 'ckks_secret.context'

                with open(ctx_path_sk_ckks, "wb") as f:
                    f.write(secret_bytes_sk)
                with open(f'{base_path}{ctx_path_sk_ckks_test}', "wb") as f:
                    f.write(secret_bytes_sk)

                context.make_context_public()

                public_bytes_pk = context.serialize(
                    save_public_key=True,
                    save_relin_keys=True,
                    save_galois_keys=True
                )

                # ctx_path_pk = "ckks_public.context"

                with open(ctx_path_pk_ckks, "wb") as f:
                    f.write(public_bytes_pk)
                with open(f'{base_path}{ctx_path_pk_ckks_test}', "wb") as f:
                    f.write(public_bytes_pk)

                return True
            except Exception as ex:
                raise Exception('[CLIENT] Failed to create CKKS Context..')

        elif schema == 'BFV':
            try:
                # Setup TenSEAL context
                context = ts.context(
                    ts.SCHEME_TYPE.BFV,
                    poly_modulus_degree=8192,
                    coeff_mod_bit_sizes=[60, 40, 40, 60],
                    plain_modulus=1032193
                )

                context.global_scale = 2 ** 40

                # Genera chiavi di automorfismo e relinearizzazione se necessario
                context.generate_galois_keys()
                context.generate_relin_keys()

                secret_bytes_sk = context.serialize(save_secret_key=True, save_public_key=True, save_relin_keys=True,
                                                 save_galois_keys=True)

                with open(ctx_path_sk_bfv, "wb") as f:
                    f.write(secret_bytes_sk)
                with open(f'{base_path}{ctx_path_sk_bfv_test}', "wb") as f:
                    f.write(secret_bytes_sk)

                context.make_context_public()

                secret_bytes_pk = context.serialize(save_public_key=True, save_relin_keys=True,
                                                 save_galois_keys=True)

                with open(ctx_path_pk_bfv, "wb") as f:
                    f.write(secret_bytes_pk)
                with open(f'{base_path}{ctx_path_pk_bfv_test}', "wb") as f:
                    f.write(secret_bytes_pk)
                return True
            except Exception as ex:
                raise Exception('Failed to create BFV Context..')
    else:
        if schema == 'CKKS':
            try:
                shutil.copy(f'{base_path}{ctx_path_sk_ckks_test}', f"{ctx_path_sk_ckks}")
                shutil.copy(f'{base_path}{ctx_path_pk_ckks_test}', f"{ctx_path_pk_ckks}")

                return True
            except Exception as ex:
                raise Exception('Failed to copy CKKS Context..')

        elif schema == 'BFV':
            try:

                shutil.copy(f'{base_path}{ctx_path_sk_bfv_test}', f"{ctx_path_sk_bfv}")
                shutil.copy(f'{base_path}{ctx_path_pk_bfv_test}', f"{ctx_path_pk_bfv}")

                return True
            except Exception as ex:
                raise Exception('Failed to copy BFV Context..')

def send_ckks_context():
    url = f"{server_url.rstrip('/')}/upload_context"
    try:
        with open(ctx_path_pk_ckks, "rb") as f:
            response = requests.post(url, files={"file": f})
        response.raise_for_status()
        # print_status(response.text)
        return True
    except requests.HTTPError as e:
        raise Exception(f"❌ HTTP error: {e} – {response.text}")
    except requests.RequestException as e:
        raise Exception(f"❌ Errore di connessione: {e}")
    except ValueError:
        raise Exception("❌ Risposta non JSON dal server")

def send_bfv_context():
    url = f"{server_url.rstrip('/')}/upload_context"
    try:
        with open(ctx_path_pk_bfv, "rb") as f:
            response = requests.post(url, files={"file": f})
        response.raise_for_status()
        # print_status(response.text)
        return True
    except requests.HTTPError as e:
        raise Exception(f"❌ HTTP error: {e} – {response.text}")
    except requests.RequestException as e:
        raise Exception(f"❌ Errore di connessione: {e}")
    except ValueError:
        raise Exception("❌ Risposta non JSON dal server")

def load_ckks_pk_context():
    with open(ctx_path_pk_ckks, "rb") as f:
        context = ts.context_from(f.read())
    return context

def load_ckks_sk_context():
    with open(ctx_path_sk_ckks, "rb") as f:
        context = ts.context_from(f.read())
    return context

def load_bfv_pk_context():
    with open(ctx_path_pk_bfv, "rb") as f:
        context = ts.context_from(f.read())
    return context

def load_bfv_sk_context():
    with open(ctx_path_sk_bfv, "rb") as f:
        context = ts.context_from(f.read())
    return context

# Lazy loading: i contesti vengono caricati solo quando necessari
_ckks_sk_context = None
_ckks_pk_context = None
_ckks_secret_key = None
_bfv_sk_context = None
_bfv_pk_context = None
_bfv_secret_key = None

def get_ckks_sk_context():
    global _ckks_sk_context
    if _ckks_sk_context is None:
        _ckks_sk_context = load_ckks_sk_context()
    return _ckks_sk_context

def get_ckks_pk_context():
    global _ckks_pk_context
    if _ckks_pk_context is None:
        _ckks_pk_context = load_ckks_pk_context()
    return _ckks_pk_context

def get_ckks_secret_key():
    global _ckks_secret_key
    if _ckks_secret_key is None:
        _ckks_secret_key = get_ckks_sk_context().secret_key()
    return _ckks_secret_key

def get_bfv_sk_context():
    global _bfv_sk_context
    if _bfv_sk_context is None:
        _bfv_sk_context = load_bfv_sk_context()
    return _bfv_sk_context

def get_bfv_pk_context():
    global _bfv_pk_context
    if _bfv_pk_context is None:
        _bfv_pk_context = load_bfv_pk_context()
    return _bfv_pk_context

def get_bfv_secret_key():
    global _bfv_secret_key
    if _bfv_secret_key is None:
        _bfv_secret_key = get_bfv_sk_context().secret_key()
    return _bfv_secret_key

def decode_decrypt_ckks(element):
    return float(element.decrypt(get_ckks_secret_key())[0])

def decode_decrypt_bfv(element):
    return float(element.decrypt(get_bfv_secret_key())[0])

def ricevi_ckks(element):
    if isinstance(element, str):
        enc_bytes = base64.b64decode(element)
    else:
        enc_bytes = element  # se già bytes
    return ts.ckks_vector_from(get_ckks_pk_context(), enc_bytes)

def ricevi_bfv(element):
    if isinstance(element, str):
        enc_bytes = base64.b64decode(element)
    else:
        enc_bytes = element  # se già bytes
    return ts.bfv_vector_from(get_bfv_pk_context(), enc_bytes)

def invia_ckks(element):
    return convert_from_ckks_bfv_to_base64(element)

def invia_bfv(element):
    return convert_from_ckks_bfv_to_base64(convert_from_object_to_bfv(element))

def convert_from_ckks_bfv_to_base64(element):
    return base64.b64encode(element.serialize()).decode("utf-8")

def convert_from_base64_to_ckks(element):
    return ts.ckks_vector_from(get_ckks_pk_context(),base64.b64decode(element))

def convert_from_base64_to_bfv(element):
    return ts.bfv_vector_from(get_bfv_pk_context(),base64.b64decode(element))

def convert_from_object_to_ckks(element):
    return ts.ckks_vector(get_ckks_pk_context(),element)

def convert_from_object_to_bfv(element):
    return ts.bfv_vector(get_bfv_pk_context(),element)

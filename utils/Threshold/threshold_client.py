import json
import os
import shutil
import tkinter as tk
import warnings
from tkinter import ttk

import pandas as pd
import requests
from tqdm import tqdm

from utils.processing_column.Manuale import detect_sensitive_columns_pandas

warnings.filterwarnings("ignore")

thresholds_file = f'threshold.json'

def load_or_create_thresholds(df: pd.DataFrame, path, calculate_thr, value_thr, dataset_name, mod, sensitive_cols_manuale=None):
    """
    Load thresholds from a JSON file or create it based on detected sensitive columns.
    """
    numeric_thr = value_thr['numeric']
    string_thr = value_thr['string']
    sensitive_cols = []

    if calculate_thr:
        modality = ['Manual', 'All_Sensitive', 'All_NOT_Sensitive']
        selected_modality = ''
        if mod in modality:
            selected_modality = mod
        else:
            selected_modality = select_type_detection(modality)

        sensitive_cols_regex = detect_sensitive_columns_pandas(df)
        df_filtered = df[[col for col in df.columns if col not in sensitive_cols_regex]]
        if 'Manual' in selected_modality:
            columns_with_types = {
                col: (
                    # 'boolean' if df[col].dtype == bool else
                    'numeric' if pd.api.types.is_numeric_dtype(df_filtered[col]) else
                    'string'
                )
                for col in tqdm(df_filtered.columns,
                                desc='[CLIENT][Thresholds]Loading Columns types',
                                total=len(df_filtered.columns),
                                leave=False,
                                disable=True)
            }

            #Test con manuale
            if sensitive_cols_manuale == None:
                sensitive_cols_manuale = select_items_from_list(columns_with_types)
            sensitive_cols = [c for c in df.columns if c in list(set(sensitive_cols_regex + sensitive_cols_manuale))]

        elif 'All_Sensitive' in selected_modality:
            sensitive_cols = [col for col in df_filtered.columns]
        elif 'All_NOT_Sensitive' in selected_modality:
            sensitive_cols = []

        template = {
            col: {
                'type': (
                    # 'boolean' if df[col].dtype == bool else
                    'numeric' if pd.api.types.is_numeric_dtype(df[col]) else
                    'string'
                ),
                'threshold': (numeric_thr if pd.api.types.is_numeric_dtype(df[col]) else
                    string_thr
                              ),
                'sensitive': col in sensitive_cols
            }
            for col in tqdm(df.columns, desc='[CLIENT][Thresholds]Loading Columns thresholds', total=len(df.columns), leave=False, disable=True)
        }

        with open(thresholds_file, 'w') as f1:
            json.dump(template, f1, indent=4)

        shutil.copy(thresholds_file,os.path.join(path,'thresholds.json'))

        del numeric_thr, string_thr
        del template, df, value_thr

    else:
        shutil.copy(os.path.join(path, 'thresholds.json'), thresholds_file)

        print('[CLIENT][Thresholds] File already exists. Modify if needed and re-run.')

def select_items_from_list(columns_with_types):
    """
    Mostra una GUI con checkbox per ogni colonna e il suo tipo.
    Ritorna le colonne selezionate.
    """
    selected_items = []

    def update_all():
        val = varAll.get()
        for col_name, var in checkbox_vars.items():
            if col_name not in ('All', 'None'):
                var.set(val)
        # deseleziona 'None' se 'All' è attivo
        if val:
            varNone.set(False)

    def update_none():
        val = varNone.get()
        if val:
            for col_name, var in checkbox_vars.items():
                if col_name not in ('All', 'None'):
                    var.set(False)
            varAll.set(False)

    def on_ok():
        for col_name, var in checkbox_vars.items():
            if col_name not in ('All', 'None') and var.get():
                selected_items.append(col_name)
        root.destroy()

    root = tk.Tk()
    root.title("Seleziona colonne sensibili")

    checkbox_vars = {}

    # Colonne normali
    for col_name, col_type in columns_with_types.items():
        display_text = f"{col_name} ({col_type})"
        var = tk.BooleanVar()
        chk = ttk.Checkbutton(root, text=display_text, variable=var)
        chk.pack(anchor='w', padx=10, pady=2)
        checkbox_vars[col_name] = var

    # Checkbox "All"
    varAll = tk.BooleanVar()
    chkAll = ttk.Checkbutton(root, text='All', variable=varAll, command=update_all)
    chkAll.pack(anchor='w', padx=10, pady=5)
    checkbox_vars['All'] = varAll

    # Checkbox "None"
    varNone = tk.BooleanVar()
    chkNone = ttk.Checkbutton(root, text='None', variable=varNone, command=update_none)
    chkNone.pack(anchor='w', padx=10, pady=2)
    checkbox_vars['None'] = varNone

    # Bottone OK
    ok_button = ttk.Button(root, text="OK", command=on_ok)
    ok_button.pack(pady=10)

    root.mainloop()

    return selected_items

def get_AllStr_columns():
    threshold = get_thr()
    return [(c, info) for c, info in threshold.items() if info.get('type') == 'string']

def get_AllNum_columns():
    threshold = get_thr()
    return [(c, info) for c, info in threshold.items() if info.get('type') == 'numeric']

def get_AllSens_columns():
    threshold = get_thr()
    return [(c, info) for c, info in threshold.items() if info.get('sensitive') == True]

def get_NoSensCol_from_list(columns):
    return [col for col in columns
        if col not in [c for c, _ in get_AllSens_columns()]
    ]

def get_SensCol_from_list(columns):
    return [col for col in columns
        if col in [c for c, _ in get_AllSens_columns()]
    ]

def get_StringSens(columns):
    return [col_df for col_df in columns if
                      col_df in [c for c, _ in get_AllSens_columns()] and
                       col_df in [c for c, _ in get_AllStr_columns()]]

def get_String_NoSens(columns):
    return [col_df for col_df in columns if
            col_df not in [c for c, _ in get_AllSens_columns()] and
            col_df in [c for c, _ in get_AllStr_columns()]]

def get_Numeric_NoSens(columns):
    return [col_df for col_df in columns if
            col_df not in [c for c, _ in get_AllSens_columns()] and
            col_df in [c for c, _ in get_AllNum_columns()]]

def get_NumericSens(columns):
    return  [col_df for col_df in columns if
                      col_df in [c for c, _ in get_AllSens_columns()] and
                       col_df in [c for c, _ in get_AllNum_columns()]]

def send_threshold(paths, server_url: str = "http://127.0.0.1:5000"):
    threshold = get_thr()

    if threshold:
        print('[CLIENT][Thresholds] Sending threshold...')
        url = f"{server_url.rstrip('/')}/save_thr"
        try:
            resp = requests.post(url, json={"thr": threshold, "path" : paths})
            resp.raise_for_status()
            data = resp.json()
            return True
        except requests.HTTPError as e:
            raise Exception(f"❌ HTTP error: {e}")
        except requests.RequestException as e:
            raise Exception(f"❌ Errore di connessione: {e}")
        except ValueError:
            raise Exception("❌ Risposta non JSON dal server")
    elif threshold is None:
        raise Exception('Cannot send thr: thresholds defined.')

def select_type_detection(modality):
        """
        Mostra una GUI con checkbox per ogni colonna e il suo tipo.
        Ritorna le colonne selezionate.
        """
        selected_items = []

        def on_ok():
            for type, var in checkbox_vars.items():
                if var.get():
                    selected_items.append(type)
            root.destroy()

        root = tk.Tk()
        root.title('[CLIENT]Seleziona modalità')

        checkbox_vars = {}

        # Colonne normali
        for type in modality:
            display_text = f"{type}"
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(root, text=display_text, variable=var)
            chk.pack(anchor='w', padx=10, pady=2)
            checkbox_vars[type] = var

        # Bottone OK
        ok_button = ttk.Button(root, text="OK", command=on_ok)
        ok_button.pack(pady=10)

        root.mainloop()

        return selected_items

def get_thr():
    with open(thresholds_file) as f:
        th = json.load(f)
    return th

def get_threshold(column):
    return [info.get('threshold') for c, info in get_thr().items() if c == column][0]


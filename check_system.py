"""
Script di Verifica Sistema - Denial Constraints Discovery
=========================================================

Questo script verifica che tutto sia configurato correttamente
prima di iniziare il workflow completo.

Uso:
    python check_system.py
"""

import sys
import os
import subprocess
import importlib.util


def check_python_version():
    """Verifica versione Python."""
    print("\n 1. Python Version")
    version = sys.version_info
    print(f"   Versione: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print(" Versione compatibile (>= 3.7)")
        return True
    else:
        print(" Versione troppo vecchia (richiesto >= 3.7)")
        return False


def check_python_packages():
    """Verifica che tutti i package Python siano installati."""
    print("\n 2. Python Packages")
    
    required_packages = [
        'numpy',
        'pandas',
        'flask',
        'requests',
        'tenseal',
        'sklearn',
        'scipy',
        'tqdm',
        'torch'  # Per SentenceTransformer
    ]
    
    missing = []
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            print(f" {package}: NON installato")
            missing.append(package)
        else:
            print(f" {package}: installato")
    
    if missing:
        print(f"\n  Installa con: pip install {' '.join(missing)}")
        return False
    return True


def check_java_version():
    """Verifica versione Java."""
    print("\n 3. Java Version")
    
    try:
        result = subprocess.run(
            ['java', '-version'],
            capture_output=True,
            text=True
        )
        
        # Java stampa su stderr
        output = result.stderr
        
        if 'version' in output:
            # Estrai numero versione
            import re
            match = re.search(r'version "(\d+)', output)
            if match:
                version = int(match.group(1))
                print(f"   Versione: {match.group(0)}")
                
                if version >= 8:
                    print("   Versione compatibile (>= 8)")
                    return True
                else:
                    print("   Versione troppo vecchia (richiesto >= 8)")
                    return False
            else:
                print(f"    Trovato ma versione non chiara: {output[:100]}")
                return True
        else:
            print("   Java non trovato")
            return False
            
    except FileNotFoundError:
        print("   Java non installato o non in PATH")
        return False


def check_fastadc():
    """Verifica che FastADC sia compilato."""
    print("\n 4. FastADC Compilation")
    
    jar_path = "FastADC/target/FastADC-1.0.jar"
    
    if os.path.exists(jar_path):
        size = os.path.getsize(jar_path)
        print(f"   FastADC JAR trovato ({size:,} bytes)")
        return True
    else:
        print(f"    FastADC JAR non trovato: {jar_path}")
        print(f"    Compila con:")
        print(f"     cd FastADC")
        print(f"     mvn clean package")
        return False


def check_files_structure():
    """Verifica struttura file del progetto."""
    print("\n 5. File Structure")
    
    required_files = {
        'server.py': 'Server principale',
        'client.py': 'Client principale',
        'utils/client_callback_server.py': 'Client callback',
        'utils/utils_Server_request.py': 'Request utilities',
        'test_dc_predicates.py': 'Script di test',
        'create_input_for_fastadc.py': 'Converter per FastADC'
    }
    
    all_found = True
    for file, desc in required_files.items():
        if os.path.exists(file):
            print(f"    {file}: trovato ({desc})")
        else:
            print(f"    {file}: MANCANTE ({desc})")
            all_found = False
    
    return all_found


def check_ports():
    """Verifica che le porte siano libere."""
    print("\n 6. Network Ports")
    
    import socket
    
    ports = {
        5000: 'Server Flask',
        5001: 'Client Callback'
    }
    
    all_free = True
    for port, desc in ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result != 0:
            print(f"    Porta {port}: libera ({desc})")
        else:
            print(f"     Porta {port}: IN USO ({desc})")
            print(f"      Potrebbe essere già avviato o altro processo")
            all_free = False
    
    return all_free


def check_dataset():
    """Verifica che ci sia almeno un dataset di esempio."""
    print("\n 7. Test Dataset")
    
    test_datasets = [
        'Test/hepatitis/hepatitis_preProcessed.csv',
        'Test/hepatitis/All_Sensitive/hepatitis_Encrypted.h5',
        'DB/hepatitis/hepatitis_preProcessed.csv'
    ]
    
    found = False
    for dataset in test_datasets:
        if os.path.exists(dataset):
            size = os.path.getsize(dataset)
            print(f"    Dataset trovato: {dataset} ({size:,} bytes)")
            found = True
            break
    
    if not found:
        print(f"     Nessun dataset di test trovato")
        print(f"    Esegui: python Initial_Setup.py")
        print(f"   Oppure posiziona dataset in: DB/")
    
    return True  # Non bloccante


def check_output_directories():
    """Verifica directory output."""
    print("\n 8. Output Directories")
    
    output_dir = "Test/hepatitis/All_Sensitive/output_N2_S0.75"
    
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"    Directory output esiste: {output_dir}")
        print(f"      File esistenti: {len(files)}")
        
        # Verifica file specifici
        expected_files = [
            'DC_Predicates.json',
            'DC_Predicate_Stats.json',
            'Matrix_Similarity.csv'
        ]
        
        any_generated = False
        for file in expected_files:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                print(f"       {file}: trovato")
                any_generated = True
            else:
                print(f"        {file}: non ancora generato (normale)")
        
        if not any_generated:
            print(f"\n      💡 I file verranno generati quando esegui il workflow")
    else:
        print(f"     Directory output non esiste (verrà creata): {output_dir}")
    
    return True  # Non bloccante


def run_connectivity_test():
    """Test veloce di connettività."""
    print("\n 9. Quick Connectivity Test")
    
    import requests
    
    # Test server
    try:
        response = requests.get('http://localhost:5000/', timeout=2)
        print(f"    Server raggiungibile (status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print(f"     Server non avviato (normale se non ancora startato)")
    except Exception as e:
        print(f"    Errore inaspettato: {e}")
    
    # Test client callback
    try:
        response = requests.post(
            'http://localhost:5001/decrypt_and_compute',
            json={'test': True},
            timeout=2
        )
        print(f"   Client callback raggiungibile (status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print(f"     Client callback non avviato (normale se non ancora startato)")
    except Exception as e:
        print(f"    Errore inaspettato: {e}")
    
    return True


def print_summary(results):
    """Stampa sommario risultati."""
    print("\n" + "=" * 70)
    print(" SOMMARIO VERIFICA")
    print("=" * 70)
    
    # Separa check critici da informativi
    critical_checks = {
        'Python Version': results.get('Python Version', False),
        'Python Packages': results.get('Python Packages', False),
        'Java': results.get('Java', False),
        'FastADC': results.get('FastADC', False),
        'File Structure': results.get('File Structure', False)
    }
    
    informative_checks = {
        'Network Ports': results.get('Network Ports', False),
        'Dataset': results.get('Dataset', False),
        'Output Directories': results.get('Output Directories', False),
        'Connectivity': results.get('Connectivity', False)
    }
    
    critical_passed = sum(critical_checks.values())
    critical_total = len(critical_checks)
    
    print(f"\n Check critici: {critical_passed}/{critical_total}")
    
    if critical_passed == critical_total:
        print("\n SISTEMA PRONTO PER L'AVVIO!")
        print("\n Check critici passati:")
        for name, passed in critical_checks.items():
            status = "corretto" if passed else "fail"
            print(f"   {status} {name}")
        
        print("\n PROSSIMI PASSI:")
        print("   1. Terminal 1: python server.py")
        print("   2. Terminal 2: python client.py")
        print("   3. Terminal 3: python test_dc_predicates.py")
        print("\n   I file di output verranno generati automaticamente!")
        
    else:
        print(f"\n  ATTENZIONE: {critical_total - critical_passed} check critici falliti")
        print("\n Check falliti:")
        for name, passed in critical_checks.items():
            if not passed:
                print(f"    fail {name}")
        
        print("\n Suggerimenti:")
        if not critical_checks.get('Python Packages'):
            print("   - Installa packages: pip install numpy pandas flask requests tenseal scikit-learn scipy tqdm torch")
        if not critical_checks.get('Java'):
            print("   - Installa Java 8+: https://adoptium.net/")
        if not critical_checks.get('FastADC'):
            print("   - Compila FastADC: cd FastADC && mvn clean package")
        
        print("\n  Risolvi i problemi critici prima di avviare il sistema.")
    
    # Mostra check informativi
    print(f"\n  Check informativi: {sum(informative_checks.values())}/{len(informative_checks)}")
    print("   (Non bloccanti - verranno soddisfatti durante l'esecuzione)")
    
    print("\n" + "=" * 70)


def main():
    """Main function."""
    print("=" * 70)
    print(" VERIFICA CONFIGURAZIONE SISTEMA")
    print("=" * 70)
    print("\nQuesta verifica controllerà che tutto sia pronto per")
    print("eseguire il workflow di DC discovery...")
    
    results = {}
    
    # Esegui tutti i check
    results['Python Version'] = check_python_version()
    results['Python Packages'] = check_python_packages()
    results['Java'] = check_java_version()
    results['FastADC'] = check_fastadc()
    results['File Structure'] = check_files_structure()
    results['Network Ports'] = check_ports()
    results['Dataset'] = check_dataset()
    results['Output Directories'] = check_output_directories()
    results['Connectivity'] = run_connectivity_test()
    
    # Stampa sommario
    print_summary(results)
    
    # Exit code basato solo su check critici
    critical_checks = [
        'Python Version',
        'Python Packages', 
        'Java',
        'FastADC',
        'File Structure'
    ]
    
    critical_passed = all(results.get(check, False) for check in critical_checks)
    
    if critical_passed:
        print("\n Puoi procedere con l'avvio del sistema!")
        sys.exit(0)
    else:
        print("\n Risolvi i problemi critici prima di procedere.")
        sys.exit(1)


if __name__ == "__main__":
    main()

import subprocess
import sys
import os
import platform
import shutil
import time

# --- Configuration: Model Registry ---
# Añade aquí nuevos modelos siguiendo el formato.
MODELS_REGISTRY = {
    "minicpm_v_2_6_8b": {
        "name": "openbmb/minicpm-v2.6:8b",
        "description": "MiniCPM-V 2.6 (8B) - Versión Estable Compatible",
    },
    "minicpm_v_4_5_8b": {
        "name": "openbmb/minicpm-v4.5:8b", 
        "description": "MiniCPM-V 4.5 (8B) - SOTA OpenBMB",
    },
    "qwen3_vl_8b": {
        "name": "qwen3-vl", 
        "description": "Qwen3-VL 8B (SOTA Razonamiento 2026)"
    },
    "internvl3_5_8b": {
        "name": "blaifa/InternVL3_5:8b",
        "description": "InternVL 3.5 (8B) - Blaifa"
    }
}

# Manejo de MSVCRT para inputs en Windows
try:
    import msvcrt
except ImportError:
    msvcrt = None

# Intento de importar psutil para métricas de RAM, manejo el error si no existe
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

# Intentar importar librerías de utilidad (requests, tqdm)
try:
    import requests
    from tqdm import tqdm
except ImportError:
    requests = None
    tqdm = None

# Habilitar secuencias ANSI en Windows (Simple & Robust)
if os.name == 'nt':
    os.system('') # Activa VT100 en Windows 10/11 sin deps extra

# --- Visual Styles ---
# Clases de colores ANSI para la interfaz de terminal
class Style:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    # Estilo visual para selección (Fondo blanco texto negro o invertido)
    SELECTED = '\033[7m' 

# --- Diagnostic Issue Class ---
# Objeto que representa un problema detectado durante el diagnóstico
class DiagnosticIssue:
    def __init__(self, description, fix_func, fix_name):
        self.description = description  # Descripción del problema para el usuario
        self.fix_func = fix_func        # La función Python que repara este problema
        self.fix_name = fix_name        # Nombre corto de la reparación para el menú
        self.label = fix_name           # Alias para el menú genérico
        self.action = fix_func          # Compatibilidad con MenuItem para ejecución directa
        self.children = []
        self.is_selected = False

# Generic Menu Item
class MenuItem:
    def __init__(self, label, action=None, description="", children=None):
        self.label = label
        self.action = action
        self.description = description
        self.children = children or []
        self.is_selected = False


# --- Configuration: Dependencies ---
# Single source of truth for required python libraries
REQUIRED_LIBS = [
    "ollama",
    "requests",
    "tqdm",
    "pillow",   # Import name: PIL
    "psutil",
    "colorama",
    "pytest",
    "pytest-mock"
]

# Mapping for import checks (Library Name -> Import Name)
# Only needed when import name differs from package name
LIB_IMPORT_MAP = {
    "pillow": "PIL",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
    "pytest-mock": "pytest_mock"
}

# --- Helper Functions ---

def clear_screen_ansi():
    """
    Limpia pantalla usando CLS en Windows para asegurar limpieza.
    """
    if os.name == 'nt':
        os.system('cls')
    else:
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()

def clear_screen():
    """Wrapper compatible."""
    clear_screen_ansi()

def read_key():
    """Lee una tecla y devuelve un código unificado (UP, DOWN, ENTER, SPACE, ESC)."""
    if os.name == 'nt' and msvcrt:
        key = msvcrt.getch()
        if key == b'\xe0':  # Teclas de dirección
            key = msvcrt.getch()
            if key == b'H': return 'UP'
            if key == b'P': return 'DOWN'
            if key == b'K': return 'LEFT'
            if key == b'M': return 'RIGHT'
        elif key == b'\r': return 'ENTER'
        elif key == b' ': return 'SPACE'
        elif key == b'\x1b': return 'ESC'
        elif key == b'\x03': raise KeyboardInterrupt # Ctrl+C
    else:
        # Fallback para input simple o Linux (si se implementa luego)
        pass
    return None

def log(msg, level="info"):
    """
    Imprime mensajes con formato y color.
    level: info, success, error, warning, step
    """
    if level == "info":
        print(f"{Style.OKCYAN} ℹ {msg}{Style.ENDC}")
    elif level == "success":
        print(f"{Style.OKGREEN} ✔ {msg}{Style.ENDC}")
    elif level == "error":
        print(f"{Style.FAIL} ✖ {msg}{Style.ENDC}")
    elif level == "warning":
        print(f"{Style.WARNING} ⚠ {msg}{Style.ENDC}")
    elif level == "step":
        print(f"\n{Style.BOLD}➤ {msg}{Style.ENDC}")

def ask_user(question, default="y"):
    """Pide confirmación al usuario (y/n)."""
    choice = input(f"{Style.WARNING}{question} [{default}/n]: {Style.ENDC}").lower().strip()
    if not choice: choice = default
    return choice.startswith("y")

def run_cmd(cmd, critical=True):
    """
    Ejecuta un comando de shell e imprime el comando en gris.
    Si falla y critical=True, pregunta al usuario si reintentar.
    """
    print(f"{Style.DIM}$ {cmd}{Style.ENDC}")
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError:
        log(f"Command failed: {cmd}", "error")
        if critical:
            if ask_user("Critical command failed. Retry?", "y"):
                return run_cmd(cmd, critical)
            return False
        return False

def get_sys_info():
    """Recopila información básica del hardware (OS, Python, CPU, GPU, RAM)."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "cpu_cores": os.cpu_count(),
        "gpu": "N/A",
        "ram": "N/A"
    }

    # RAM via psutil
    if psutil:
        try:
            mem = psutil.virtual_memory()
            info["ram"] = f"{mem.total / (1024**3):.1f} GB"
        except:
             info["ram"] = "Unknown"
    else:
        info["ram"] = "Unknown (psutil missing)"

    # GPU via nvidia-smi
    try:
        res = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True)
        info["gpu"] = res.decode("utf-8").strip()
    except:
        info["gpu"] = "Not Detected (CPU Mode)"
    
    return info

def check_ollama():
    """Verifica si 'ollama' está instalado y el servicio corriendo."""
    try:
        subprocess.check_call("ollama --version", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def pull_ollama_model(model_key):
    """Descarga (pull) un modelo de Ollama."""
    if model_key not in MODELS_REGISTRY:
        log(f"Model key '{model_key}' not found in registry.", "error")
        return False
    
    entry = MODELS_REGISTRY[model_key]
    model_tag = entry["name"]
    
    log(f"Pulling {model_tag} ({entry['description']})...", "step")
    
    # Check if ollama is running first (simple check)
    if not check_ollama():
        log("Ollama binary not found or not in PATH.", "error")
        log("Please install Ollama from https://ollama.com/", "warning")
        return False

    try:
        # Use subprocess to stream output properly if possible, or just wait
        # "ollama pull" writes to stderr usually for progress bars
        subprocess.check_call(f"ollama pull {model_tag}", shell=True)
        log(f"Successfully pulled {model_tag}", "success")
        return True
    except subprocess.CalledProcessError:
        log(f"Failed to pull {model_tag}. Is Ollama server running?", "error")
        return False

def list_test_files():
    """Escanea la carpeta tests/ y devuelve los archivos .py válidos."""
    test_dir = "tests"
    if not os.path.exists(test_dir):
        return []
    
    files = [f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")]
    return sorted(files)

def manage_models_menu_ui():
    """Menú dinámico para gestionar descargas de modelos OLLAMA."""
    def header():
        print_banner()
        print(f"{Style.BOLD} OLLAMA MODEL MANAGER {Style.ENDC}")
        print(" Select a model to pull:")

    while True:
        options = []
        
        # Check if ollama is running
        ollama_status = "RUNNING" if check_ollama() else "NOT FOUND/STOPPED"
        ollama_color = Style.OKGREEN if ollama_status == "RUNNING" else Style.FAIL
        
        # Try to get list of installed models
        installed_models = []
        if check_ollama():
            try:
                res = subprocess.check_output("ollama list", shell=True).decode()
                installed_models = res
            except:
                pass

        for key, data in MODELS_REGISTRY.items():
            model_tag = data["name"]
            is_installed = model_tag in installed_models
            
            status_icon = "✔ PULLED" if is_installed else "☁ PULL"
            status_color = Style.OKGREEN if is_installed else Style.OKBLUE
            
            label = f" {data['name']:<30} | {status_color}{status_icon}{Style.ENDC}"
            
            # Action wrapper using closure for key capture
            action = lambda k=key: (pull_ollama_model(k), input(f"\n{Style.DIM}Press Enter...{Style.ENDC}"))
            
            options.append(MenuItem(label, action, description=data["description"]))

        options.append(MenuItem(" Back", lambda: "BACK"))
    
        choice = interactive_menu(options, header_func=header)
        
        if not choice:
            break
            
        # choice es un objeto MenuItem, hay que ejecutar su acción
        if hasattr(choice, 'action') and callable(choice.action):
            clear_screen_ansi()
            res = choice.action()
            if res == "BACK":
                break    

def run_tests_menu():
    """Menú para ejecutar tests (Unitarios y Smoke Tests)."""
    
    def run_all_unit_tests():
        log("Running All Unit Tests...", "step")
        run_cmd("uv run python -m pytest tests/")
        input(f"\n{Style.DIM}Press Enter to return...{Style.ENDC}")

    def run_specific_test():
        files = list_test_files()
        if not files:
            log("No tests found in tests/ folder.", "warning")
            time.sleep(1)
            return

        test_opts = [MenuItem(f) for f in files]
        test_opts.append(MenuItem(" Cancel", lambda: None))

        def t_header():
             print_banner()
             print(f"{Style.BOLD} SELECT TEST FILE {Style.ENDC}")

        selection = interactive_menu(test_opts, header_func=t_header)
        
        if selection and selection.label != " Cancel":
             # selected item
             fname = selection.label
             log(f"Running {fname}...", "step")
             # Try running as pytest target first
             run_cmd(f"uv run python -m pytest tests/{fname}")
             input(f"\n{Style.DIM}Finished. Press Enter...{Style.ENDC}")

        # Interactive Model Selection
        # Create descriptive labels 
        model_opts = []
        
        # Add registry models
        for key, val in MODELS_REGISTRY.items():
             model_opts.append(MenuItem(val["name"]))

        model_opts.append(MenuItem(" Cancel", lambda: None))

        def m_header():
             print_banner()
             print(f"{Style.BOLD} SELECT INFERENCE MODEL {Style.ENDC}")

        selection = interactive_menu(model_opts, header_func=m_header)
        
        if selection and selection.label.strip() != "Cancel":
            model_tag = selection.label
            
            log(f"Launching Inference with {model_tag}...", "step")
            try:
                # Pass selected model tag as argument to the script
                subprocess.check_call(["uv", "run", "python", "src/scripts/test_inference.py", "--model_path", model_tag])
            except subprocess.CalledProcessError:
                log("Smoke test failed (Exit Code 1). Check output above.", "error")
            input(f"\n{Style.DIM}Press Enter to return...{Style.ENDC}")

    def run_smoke_test_wrapper():
        """Wrapper para lanzar el smoke test interactivo."""
        # Reutilizamos la lógica de selección de modelo definida dentro de run_specific_test
        # O mejor, la movemos a una función helper.
        # Por simplicidad y evitar refactor mayor ahora, copiamos la lógica o llamamos a una shared.
        
        # Simplificación: copiamos lógica de selección de modelo aquí.
        model_opts = []
        for key, val in MODELS_REGISTRY.items():
             model_opts.append(MenuItem(val["name"]))
        model_opts.append(MenuItem(" Cancel", lambda: None))

        def m_header():
             print_banner()
             print(f"{Style.BOLD} SELECT INFERENCE MODEL {Style.ENDC}")

        selection = interactive_menu(model_opts, header_func=m_header)
        
        if selection and selection.label.strip() != "Cancel":
            model_tag = selection.label
            log(f"Launching Inference with {model_tag}...", "step")
            try:
                subprocess.check_call(["uv", "run", "python", "src/scripts/test_inference.py", "--model_path", model_tag])
            except subprocess.CalledProcessError:
                log("Smoke test failed (Exit Code 1). Check output above.", "error")
            input(f"\n{Style.DIM}Press Enter to return...{Style.ENDC}")


    def header():
        print_banner()
        print(f"{Style.BOLD} TEST & MODEL MANAGER {Style.ENDC}")

    options = [
        MenuItem(" Run All Unit Tests (pytest)", run_all_unit_tests),
        MenuItem(" Run Specific Test File...", run_specific_test),
        MenuItem(" Run Smoke Test (Inference Demo)", run_smoke_test_wrapper),
        MenuItem(" Manage/Pull Ollama Models...", manage_models_menu_ui),
        MenuItem(" Return to Main Menu", lambda: "BACK")
    ]

    while True:
        choice = interactive_menu(options, header_func=header, multi_select=False)
        if choice == "BACK" or not choice:
            break
        
        # Ensure choice is a single item, not a list
        if isinstance(choice, list):
            choice = choice[0] if len(choice) > 0 else None
        
        if choice and hasattr(choice, 'action') and callable(choice.action):
            clear_screen_ansi()
            res = choice.action()
            if res == "BACK": break

def print_banner():
    """Muestra el banner principal con el estado del sistema."""
    clear_screen()
    info = get_sys_info()
    
    print(f"{Style.HEADER}{Style.BOLD}")
    print("╔═════════════════════════════════════════════════════════════════╗")
    print("║              🩺 TFG VLM Medical - Manager Tool v4.0             ║")
    print("╠═════════════════════════════════════════════════════════════════╣")
    print(f"║ {Style.OKCYAN}OS     : {info['os']:<46}{Style.HEADER}         ║")
    print(f"║ {Style.OKCYAN}Python : {info['python']:<46}{Style.HEADER}         ║")
    print(f"║ {Style.OKCYAN}CPU    : {info['cpu_cores']} Cores{' ':<39}{Style.HEADER}        ║")
    print(f"║ {Style.OKCYAN}RAM    : {info['ram']:<46}{Style.HEADER}         ║")
    print(f"║ {Style.OKCYAN}GPU    : {info['gpu']:<46}{Style.HEADER}         ║")
    print("╚═════════════════════════════════════════════════════════════════╝")

def restart_program():
    """Reinicia el script actual de forma limpia."""
    print(f"\n{Style.BOLD}{Style.WARNING}🔴 CRITICAL UPDATE DETECTED (TORCH/CORE) 🔴{Style.ENDC}")
    print(f"{Style.WARNING}The application must restart to load the new libraries correctly.{Style.ENDC}")
    print(f"{Style.WARNING}Restarting in 2 seconds...{Style.ENDC}")
    time.sleep(2)
    
    # Restore terminal state before restart (Evita que la terminal se 'rompa')
    print("\033[?25h", end="") # Show cursor explicitly
    sys.stdout.flush()
    
    python = sys.executable
    # En Windows, os.execl no reemplaza realmente el proceso, lo que causa el "ghosting".
    # Usamos subprocess para lanzar el hijo y esperar a que termine, manteniendo la terminal ocupada.
    try:
        subprocess.check_call([python] + sys.argv)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error restarting: {e}")
    
    sys.exit(0)


# --- Core Utility Functions ---

def detect_gpu():
    """Detecta si hay una GPU NVIDIA disponible usando nvidia-smi."""
    try:
        subprocess.check_call("nvidia-smi", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def check_uv():
    """Verifica si 'uv' está instalado y accesible en el PATH."""
    try:
        subprocess.check_call("uv --version", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def create_project_structure(verbose=True):
    """Crea la estructura de carpetas necesaria si no existe."""
    folders = [
        "data/raw",
        "data/processed",
        "models",
        "src/preprocessing",
        "src/inference",
        "notebooks"
    ]
    
    created = 0
    for folder in folders:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
                if verbose: log(f"Created missing directory: {folder}", "success")
                created += 1
            except Exception as e:
                log(f"Failed to create {folder}: {e}", "error")
    
    if verbose and created == 0:
        log("All directories verified.", "success")

# --- Modular Fixers ---
# Estas funciones son llamadas automáticamente por el Smart Fix Menu

def fix_folders():
    """Reparación: Regenera carpetas faltantes."""
    log("Regenerating folder structure...", "step")
    create_project_structure(verbose=True)

def fix_uv():
    """Reparación: Instala la herramienta uv."""
    log("Installing 'uv' package manager...", "step")
    run_cmd("pip install uv")

def fix_torch():
    """Reparación: Reinstala PyTorch detectando CUDA 12.1 o CPU."""
    log("Re-installing PyTorch (Auto-detecting GPU)...", "step")
    use_gpu = detect_gpu()
    if use_gpu:
        run_cmd("uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        run_cmd("uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")

def fix_libs(libs_to_install=None):
    """Reparación: Reinstala una lista de librerías en modo Soft (sin romper dependencias)."""
    if libs_to_install is None:
        # Usar la lista maestra si no se especifican libs
        libs_to_install = REQUIRED_LIBS
    
    # Aseguramos que sea lista
    if isinstance(libs_to_install, str):
        libs_to_install = [libs_to_install]

    libs_str = " ".join(libs_to_install)
    log(f"Installing libraries: {libs_str}...", "step")
    # Eliminamos --no-deps para asegurar que se instalen las dependencias faltantes (ej: requests)
    run_cmd(f"uv pip install --force-reinstall {libs_str}")



# --- Diagnostic System ---

def perform_diagnostics():
    """
    Ejecuta comprobaciones del sistema.
    Retorna:
    - report: Lista de tuplas (Nombre, Estado, ColorStatus) para mostrar.
    - issues: Lista de objetos DiagnosticIssue con soluciones vinculadas.
    """
    report = []
    issues = []
    
    # 1. Folder Structure Check
    folders = ["data/raw", "data/processed", "models", "src/preprocessing", "src/inference", "notebooks"]
    missing_folders = False
    for folder in folders:
        if os.path.exists(folder):
             report.append((f"Dir: {folder}", "Exists", "OK"))
        else:
             report.append((f"Dir: {folder}", "MISSING", "FAIL"))
             missing_folders = True
    
    if missing_folders:
        issues.append(DiagnosticIssue("Missing Folders", fix_folders, "Regenerate Folders"))

    # 2. UV Method Check
    if check_uv():
        report.append(("Tool: uv", "Installed", "OK"))
    else:
        report.append(("Tool: uv", "MISSING", "FAIL"))
        issues.append(DiagnosticIssue("Missing uv", fix_uv, "Install uv"))



    # 4. Key Libraries
    # Verificamos importación de librerías clave una por una
    for pkg_name in REQUIRED_LIBS:
        # Determine import name
        import_name = LIB_IMPORT_MAP.get(pkg_name, pkg_name)
        
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "Unknown")
            report.append((f"Lib: {pkg_name}", version, "OK"))
        except ImportError:
            report.append((f"Lib: {pkg_name}", "MISSING", "FAIL"))
            fix_lambda = lambda p=[pkg_name]: fix_libs(p)
            issues.append(DiagnosticIssue(f"Missing {pkg_name}", fix_lambda, f"Install {pkg_name}"))
        except Exception as e:
            report.append((f"Lib: {pkg_name}", "ERROR", "FAIL"))
            fix_lambda = lambda p=[pkg_name]: fix_libs(p)
            issues.append(DiagnosticIssue(f"Error {pkg_name}", fix_lambda, f"Reinstall {pkg_name}"))

    # Ollama Service Check
    if check_ollama():
        report.append(("Service: Ollama", "Running", "OK"))
    else:
        report.append(("Service: Ollama", "NOT DETECTED", "FAIL"))
        issues.append(DiagnosticIssue("Ollama not finding", lambda: log("Install Ollama from https://ollama.com", "warning"), "Install Ollama"))

    # Ollama Python Lib
    try:
        import ollama
        report.append(("Lib: ollama", "Installed", "OK"))
    except ImportError:
        report.append(("Lib: ollama", "MISSING", "FAIL"))
        issues.append(DiagnosticIssue("Missing ollama lib", lambda: fix_libs("ollama"), "Install olama lib"))

    # 5. Storage
    free_gb = shutil.disk_usage(".").free / (1024**3)
    if free_gb < 10: 
        report.append(("Disk Space", f"{free_gb:.1f} GB", "WARN"))
    else:
        report.append(("Disk Space", f"{free_gb:.1f} GB (Free)", "OK"))

    return report, issues


def print_report_table(report):
    """Imprime la tabla de diagnóstico de forma formateada."""
    print(f"\n{Style.HEADER}┌──────────────────────────┬───────────────────────────────┬──────┐{Style.ENDC}")
    print(f"{Style.HEADER}│ Component                │ Status/Value                  │ Res  │{Style.ENDC}")
    print(f"{Style.HEADER}├──────────────────────────┼───────────────────────────────┼──────┤{Style.ENDC}")
    
    for name, value, status in report:
        color = Style.OKGREEN if status == "OK" else (Style.WARNING if status == "WARN" else Style.FAIL)
        # Limpiar nombres para visualización
        display_name = name.replace("cv2", "OpenCV").replace("llama_cpp", "LlamaCPP")
        print(f"│ {display_name:<24} │ {str(value):<29} │ {color}{status:<4}{Style.ENDC} │")

    print(f"{Style.HEADER}└──────────────────────────┴───────────────────────────────┴──────┘{Style.ENDC}")


def interactive_menu(options, header_func=None, multi_select=False, info_text=""):
    """
    Menú interactivo genérico controlado por teclado.
    Soporta circularidad y navegación por sub-niveles.
    """
    current_row = 0
    in_sub_nav = False
    
    # Hide cursor
    print("\033[?25l", end="")
    
    # Limpieza inicial completa con CLS
    if os.name == 'nt':
        os.system('cls')
    else:
        sys.stdout.write("\033[H\033[2J")

    try:
        while True:
            # 1. Construir lista plana de filas visibles
            flat_rows = []
            for opt in options:
                # El elemento principal
                flat_rows.append({'obj': opt, 'level': 0})
                # Hijos (siempre visibles según el usuario)
                if hasattr(opt, 'children') and opt.children:
                    for child in opt.children:
                        flat_rows.append({'obj': child, 'level': 1, 'parent': opt})

            # 2. Asegurar circularidad inicial y clamping
            if not flat_rows: return None
            current_row = current_row % len(flat_rows)

            # 3. Renderizado
            clear_screen_ansi()
            if header_func: header_func()
            if info_text: print(f"{info_text}\n")
            
            # Ayuda contextual
            if in_sub_nav:
                print(f"{Style.DIM} [SUB-NAV] SPACE: Seleccionar/Deseleccionar, ESC: Volver atrás, ENTER: Confirmar todo.{Style.ENDC}")
            else:
                print(f"{Style.DIM} Arriba/Abajo: Navegar, SPACE: Entrar en Submenú / Seleccionar, ENTER: Confirmar Todo.{Style.ENDC}")

            # Viewport Calculation (Keep cursor centered)
            term_height = 24
            if shutil:
                term_height = shutil.get_terminal_size().lines
            
            # Lines reserved roughly for: (Banner/Header ~10) + (Info Text ~2) + (Footer ~3) + Space
            reserved_lines = 15 
            max_visible_items = max(5, term_height - reserved_lines)
            
            # Calculate scroll offset to keep current_row centered
            half_view = max_visible_items // 2
            start_row = max(0, current_row - half_view)
            end_row = start_row + max_visible_items
            
            # Adjustment if we are near the end (but for circular menus this is tricky, let's keep it simple mostly linear view)
            if end_row > len(flat_rows):
                end_row = len(flat_rows)
                start_row = max(0, end_row - max_visible_items)

            visible_slice = flat_rows[start_row:end_row]

            # Determine current level for filtering indicators
            cur_level = flat_rows[current_row]['level']
            
            # Top Indicator
            if start_row > 0:
                hidden_above = sum(1 for r in flat_rows[:start_row] if r['level'] == cur_level)
                msg = f"▲ ({hidden_above}) ▲" if hidden_above > 0 else "▲"
                print(f"  {Style.DIM}{msg}{Style.ENDC}")
            else:
                print(" ")

            # Render Loop
            for i, row in enumerate(visible_slice):
                idx = start_row + i # Actual index in flat_rows
                is_selected_row = (idx == current_row)
                item = row['obj']
                level = row['level']
                
                # Checkbox
                checkbox = ""
                if multi_select:
                    is_checked = getattr(item, 'is_selected', False)
                    checkbox = f"[{'x' if is_checked else ' '}] "
                
                pointer = "👉" if is_selected_row else "  "
                indent = "    " * level
                suffix = " >" if level == 0 and hasattr(item, 'children') and item.children else ""
                label = getattr(item, 'label', getattr(item, 'fix_name', str(item)))
                
                line_content = f"{pointer} {indent}{checkbox}{label}{suffix}"
                if is_selected_row:
                    print(f"{Style.SELECTED} {line_content:<60} {Style.ENDC}")
                else:
                    print(f"  {line_content}")
            
            # Bottom Indicator
            if end_row < len(flat_rows):
                hidden_below = sum(1 for r in flat_rows[end_row:] if r['level'] == cur_level)
                msg = f"▼ ({hidden_below}) ▼" if hidden_below > 0 else "▼"
                print(f"  {Style.DIM}{msg}{Style.ENDC}")
            else:
                print(" ")

            # Footer
            print("\n" + "─"*65)
            if multi_select:
                # Contar seleccionados totales
                res = []
                def collect(items):
                    for i in items:
                        if i.is_selected: res.append(i)
                        if hasattr(i, 'children'): collect(i.children)
                collect(options)
                print(f" {Style.BOLD}[ENTER]: Confirmar selección ({len(res)} elementos).{Style.ENDC}")
            else:
                print(f" {Style.BOLD}[ENTER]: Seleccionar opción.{Style.ENDC}")
            
            # 4. Input Handling
            key = read_key()
            if key == 'UP':
                if in_sub_nav:
                     # Rotar solo dentro del submenú actual (hijos del mismo padre)
                    current_parent = flat_rows[current_row].get('parent')
                    if current_parent:
                        # Encontrar rango de índices de este padre
                        siblings = [i for i, r in enumerate(flat_rows) if r.get('parent') == current_parent]
                        if siblings:
                            # Encontrar índice relativo actual
                            try:
                                rel_idx = siblings.index(current_row)
                                new_rel_idx = (rel_idx - 1) % len(siblings)
                                current_row = siblings[new_rel_idx]
                            except ValueError:
                                pass # Should not happen
                else:
                    # Navegación nivel superior
                    next_row = (current_row - 1) % len(flat_rows)
                    while flat_rows[next_row]['level'] > 0:
                        next_row = (next_row - 1) % len(flat_rows)
                    current_row = next_row

            elif key == 'DOWN':
                if in_sub_nav:
                    # Rotar solo dentro del submenú actual
                    current_parent = flat_rows[current_row].get('parent')
                    if current_parent:
                        siblings = [i for i, r in enumerate(flat_rows) if r.get('parent') == current_parent]
                        if siblings:
                            rel_idx = siblings.index(current_row)
                            new_rel_idx = (rel_idx + 1) % len(siblings)
                            current_row = siblings[new_rel_idx]
                else:
                    # Navegación nivel superior
                    next_row = (current_row + 1) % len(flat_rows)
                    while flat_rows[next_row]['level'] > 0:
                        next_row = (next_row + 1) % len(flat_rows)
                    current_row = next_row
            
            elif key == 'SPACE' and multi_select:
                row = flat_rows[current_row]
                item = row['obj']
                
                if row['level'] == 0:
                     # Es un padre -> Entrar en Submenú (Si tiene hijos)
                    if hasattr(item, 'children') and item.children:
                        in_sub_nav = True
                        # Buscar primer hijo
                        for i, r in enumerate(flat_rows):
                            if r['level'] == 1 and r.get('parent') == item:
                                current_row = i
                                break
                    else:
                        # Si no tiene hijos, actúa como toggle normal
                        item.is_selected = not item.is_selected
                else:
                    # Es un hijo -> Toggle selección
                    item.is_selected = not item.is_selected
            
            elif key == 'ENTER':
                # ENTER SIEMPRE confirma todo (o selecciona si no es multi)
                row = flat_rows[current_row]
                item = row['obj']
                
                if multi_select:
                    selected_objs = []
                    def collect(items):
                        for i in items:
                            if i.is_selected: selected_objs.append(i)
                            if hasattr(i, 'children'): collect(i.children)
                    collect(options)
                    return selected_objs
                
                # Single select logic usually returns the item
                return item
            
            elif key == 'ESC':
                if in_sub_nav:
                    in_sub_nav = False
                    # Volver al padre
                    if flat_rows[current_row]['level'] > 0:
                        parent = flat_rows[current_row].get('parent')
                        if parent:
                            for i, r in enumerate(flat_rows):
                                if r['obj'] == parent:
                                    current_row = i
                                    break
                else:
                    return None
    finally:
        # Show cursor again
        print("\033[?25h", end="")


def smart_fix_menu(issues, report=None):
    """
    Menú gráfico para seleccionar arreglos.
    """
    # 1. Preparar lista única
    seen_funcs = set()
    display_options = []
    
    for issue in issues:
        # Usamos fix_func como clave única de la operación
        func_key = issue.fix_func
        if func_key not in seen_funcs:
            # Asignamos label para el menú (Visual)
            issue.label = issue.description 
            # Asignamos la acción directa para que interactive_menu pueda ejecutarla si se deseara (aunque aquí lo hacemos en batch)
            issue.action = issue.fix_func
            
            display_options.append(issue)
            seen_funcs.add(func_key)
    
    # Header dinámico (Reporte + Lista de errores)
    def draw_header():
        if report:
             print(f"\n{Style.BOLD}--- SYSTEM DIAGNOSTICS (CACHED) ---{Style.ENDC}")
             print_report_table(report)
        print(f"{Style.BOLD}{Style.FAIL} DIAGNOSTIC ISSUES DETECTED ({len(issues)}) {Style.ENDC}")
        # Listar errores brevemente
        for i in issues:
            print(f" {Style.FAIL}●{Style.ENDC} {i.description}")

    # 2. Ejecutar menú
    selected_fixes = interactive_menu(
        display_options, 
        header_func=draw_header,
        multi_select=True,
        info_text=""
    )
    
    # 3. Procesar
    if selected_fixes:
        clear_screen_ansi()
        print(f"\n{Style.BOLD} APPLYING FIXES... {Style.ENDC}")
        time.sleep(0.5)
        
        # Asegurar que iteramos sobre una lista (compatibilidad linter)
        fixes_list = selected_fixes if isinstance(selected_fixes, list) else [selected_fixes]
        
        # Sort Fixes: Standard fixes first, Torch/Llama/Heavy fixes LAST to prevent downgrades
        def sort_key(fix_item):
            name = fix_item.fix_func.__name__
            if "fix_torch" in name: return 2
            if "fix_llama" in name: return 2
            return 1 # Standard fixes (libs, uv, folders)

        fixes_list.sort(key=sort_key)

        # Verificar si necesitamos reiniciar (si se toca Torch)
        restart_required = False

        for fix in fixes_list:
            # fix es un objeto DiagnosticIssue
            # Check for restart trigger before running
            if fix.fix_func and "fix_torch" in fix.fix_func.__name__:
                restart_required = True
            
            fix.action() 
        
        if restart_required:
            restart_program()
        
        log("All fixes applied. Refreshing diagnostics...", "success")
        time.sleep(1.0)
        return True # Indicate rerun needed
    
    return False # Cancelled 


def run_diagnostics_ui():
    """Wrapper UI para ejecutar diagnósticos y mostrar resultados."""
    while True:
        clear_screen()
        print(f"\n{Style.BOLD}--- SYSTEM DIAGNOSTICS & VERIFICATION ---{Style.ENDC}")
        
        report, issues = perform_diagnostics()

        # Print Report Table
        print_report_table(report)
        
        if issues:
            print(f"\n{Style.FAIL}Issues found! Entering Smart Fix Menu...{Style.ENDC}")
            time.sleep(1)
            # Pass report so menu can re-print it
            should_rerun = smart_fix_menu(issues, report)
            if should_rerun:
                continue # Re-run checks and refresh table
            else:
                break # Cancelled by user
        else:
            print(f"\n{Style.OKGREEN}System looks healthy!{Style.ENDC}")
            input(f"\n{Style.DIM}Press Enter to return...{Style.ENDC}")
            break # Return to main menu

# --- Install & Menus ---

def reinstall_library_menu():
    """Menú manual para forzar reinstalaciones limpias."""
    def header():
        print_banner()
        print(f"{Style.BOLD} REINSTALL LIBRARIES {Style.ENDC}")
        print(" Select libraries to force re-install (clean install).")

    # Definición de librerías para el bloque core (Lightweight)
    # Definición de librerías para el bloque core (Lightweight)
    core_children = [MenuItem(lib) for lib in REQUIRED_LIBS]

    # Opciones principales
    opts = [
        MenuItem("Reinstall Core Dependencies", children=core_children),
        MenuItem("Reinstall 'uv' Tool", lambda: fix_uv())
    ]
    
    selected = interactive_menu(opts, header_func=header, multi_select=True)
    
    if selected:

        clear_screen_ansi()
        
        # Agrupar librerías core para instalarlas en un solo comando uv
        selected_core_libs = [s.label for s in selected if s in core_children]
        other_tasks = [s for s in selected if s not in core_children and callable(s.action)]
        
        if selected_core_libs:
            fix_libs(selected_core_libs)
            
        for task in other_tasks:
            log(f"Ejecutando {task.label}...", "step")
            task.action()
        
        # Detectar necesidad de reinicio manual para este menú
        restart_needed = False
        # Si se instalaron librerías core, recomendamos reinicio
        if selected_core_libs:
            restart_needed = True


        if restart_needed:
            restart_program()

        log("Operaciones completadas.", "success")
        input(f"\n{Style.DIM}Presione Enter para continuar...{Style.ENDC}")

def perform_install(full_reinstall=False):
    """
    Flujo de instalación inicial o reseteo de fábrica.
    Crea el venv y lanza las instalaciones secuenciales.
    """
    force_flags = ""

    if full_reinstall:
        if os.path.exists(".venv"):
            # Check if running inside the venv we are trying to delete
            current_exe = os.path.normpath(sys.executable)
            venv_path = os.path.abspath(".venv")
            
            if venv_path in current_exe:
                log("⚠️  Cannot delete active .venv (File in use).", "warning")
                log("🔄 Switching to 'Force Reinstall' mode (Overwriting libraries)...", "step")
                force_flags = "--force-reinstall"
                time.sleep(2)
            else:
                log("Removing existing environment (.venv)...", "warning")
                try:
                    shutil.rmtree(".venv")
                    log("Environment removed.", "success")
                    time.sleep(1)
                except Exception as e:
                    log(f"Failed to delete .venv: {e}", "error")
                    log("Please delete the folder manually.", "warning")
                    return

    # 1. Project Structure
    create_project_structure(verbose=False)

    # 2. UV Check
    if not check_uv():
        log("uv not found. Attempting to install via pip...", "step")
        if not run_cmd("pip install uv"):
            log("Could not install uv. Please install it manually.", "error")
            return

    # 3. Hardware Strategy
    use_gpu = detect_gpu()
    if not use_gpu:
        log("No NVIDIA GPU detected. Installing in CPU mode.", "warning")
    else:
        log("NVIDIA GPU detected. Using High Performance configuration.", "success")
    
    # 4. Virtual Env
    log("Configuring Virtual Environment...", "step")
    if not os.path.exists(".venv"):
        run_cmd("uv venv .venv --python 3.12")
    else:
        log(".venv exists.", "info")

    # 5. Libraries
    log("Installing dependencies...", "step")
    # Only light dependencies for Ollama workflow + Testing
    # Removed: torch torchvision torchaudio transformers accelerate protobuf scipy opencv-python bitsandbytes sentencepiece
    light_libs = " ".join(REQUIRED_LIBS)
    run_cmd(f"uv pip install {force_flags} {light_libs}")

def show_menu():
    """Bucle principal del menú."""
    
    def header():
        print_banner()
        print(f"{Style.BOLD} MAIN MENU {Style.ENDC}")

    # Actions wrappers
    def action_diag(): run_diagnostics_ui()
    def action_reinstall(): reinstall_library_menu()
    def action_regen(): 
        create_project_structure(verbose=True)
        input(f"\n{Style.DIM}Press Enter to return...{Style.ENDC}")
    def action_reset():
        if ask_user("This will delete .venv and reinstall everything. Sure?"):
            perform_install(full_reinstall=True)

    options = [
        MenuItem(" Run System Diagnostics", action_diag),
        MenuItem(" Tests & Models Manager", run_tests_menu), # New Menu
        MenuItem(" Manual Reinstall Menu", action_reinstall),
        MenuItem(" Regenerate Folders", action_regen),
        MenuItem(" Factory Reset", action_reset),
        MenuItem(" Exit", lambda: sys.exit(0))
    ]

    while True:
        # Single selection menu
        choice = interactive_menu(options, header_func=header, multi_select=False)
        
        if choice:
            # Si el linter detecta una lista (por ser el retorno posible del menú genérico), 
            # nos aseguramos de tratarlo como un solo objeto.
            item = choice[0] if isinstance(choice, list) else choice
            
            # Execute action
            # We clear before action to give space for logs
            clear_screen_ansi()
            item.action()
            # Loop continues...
        else:
            # ESC or Enter on nothing -> Exit
            clear_screen_ansi()
            print("Goodbye!")
            sys.exit(0)

def main():
    venv_dir = ".venv"
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe") if os.name == 'nt' else os.path.join(venv_dir, "bin", "python")
    
    # Check if we are running inside the venv
    # We compare the current executable path with the expected venv python path
    running_in_venv = False
    try:
        if os.path.exists(venv_python) and os.path.samefile(sys.executable, venv_python):
            running_in_venv = True
    except:
        pass # Handle potential errors in path comparison

    if not running_in_venv:
        # If venv exists, switch to it immediately
        if os.path.exists(venv_python):
            # print(f"🔄 Switching to virtual environment: {venv_python}")
            try:
                subprocess.check_call([venv_python] + sys.argv)
                sys.exit(0)
            except Exception as e:
                print(f"❌ Failed to switch to venv: {e}")
                sys.exit(1)
        
        # If venv does NOT exist (First Run)
        print_banner()
        log("First run detected. Starting setup...", "info")
        perform_install()
        
        # After install, we MUST restart to load the new venv
        if os.path.exists(venv_python):
            log("Restarting in new environment...", "success")
            time.sleep(1)
            subprocess.check_call([venv_python] + sys.argv)
            sys.exit(0)
        else:
            log("Error: .venv was not created properly.", "error")
            sys.exit(1)

    # If we are here, we are running INSIDE the venv (or failed to switch)
    show_menu()

if __name__ == "__main__":
    main()

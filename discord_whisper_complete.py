# Discord Whisper Overlay
# Versión: 1.0.0
# Fecha: 19 de abril de 2025
# Autor: GitHub Copilot y usuario
# Descripción: Overlay para transcripción en tiempo real de audio de micrófono y Discord
# Licencia: MIT

# Este archivo contiene la versión final y completa del Discord Whisper Overlay,
# con todas las mejoras implementadas, incluyendo:
# - Interfaz en modo chat
# - Detección de audio mejorada
# - Soporte para GPU (CUDA)
# - Eliminación de alucinaciones
# - Click-through para no interferir con otras aplicaciones
# - Timestamps de mensajes
# - Soporte para español optimizado

import os
import sys
import threading
import time
import ctypes
import traceback
import logging
from datetime import datetime
from collections import deque
import random
import torch
import re

import numpy as np
import pyaudio
import whisper
from PyQt5 import QtWidgets, QtCore, QtGui

# Intentar importar psutil para estadísticas del sistema
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Nota: El módulo psutil no está instalado. Las estadísticas de rendimiento serán limitadas.")
    print("      Para instalarlo ejecuta: pip install psutil")

# Configurar CUDA y PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Usar la primera GPU
# Configurar PyTorch para optimizar para RTX
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# CONFIGURACIÓN DE LOGS
LOGS_ENABLED = True         # Activar/desactivar sistema de logs
LOG_FILE = "whisper_log"    # Nombre base del archivo de log (se añadirá fecha)
LOG_FOLDER = "logs"         # Carpeta donde se guardarán los logs
MAX_LOG_FILES = 10          # Número máximo de archivos de log a mantener
LOG_LEVEL = logging.INFO    # Nivel de detalle: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"  # Formato del log
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"  # Formato de fecha/hora en logs
LOG_STATS_INTERVAL = 60     # Intervalo en segundos para estadísticas periódicas
LOG_TO_CONSOLE = True       # Mostrar logs en consola además de archivo
LOG_TRANSCRIPTIONS = True   # Registrar todas las transcripciones en el log
LOG_PERFORMANCE = True      # Registrar estadísticas de rendimiento

# CONFIGURACIÓN DE HARDWARE
USE_GPU = True           # Usar GPU para aceleración si está disponible
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
FORCE_CPU = False        # Forzar CPU incluso si hay GPU disponible
HALF_PRECISION = True    # Usar precisión media (FP16) para mayor velocidad en GPU
CUDA_VISIBLE_DEVICES = "0"  # Índice de la GPU a usar (0 es la primera GPU)
OPTIMIZE_FOR_RTX = True     # Optimizaciones especiales para tarjetas RTX
DYNAMIC_MEMORY_ALLOCATION = True  # Permitir asignación dinámica de memoria en GPU

# CONFIGURACIÓN DE AUDIO
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
BUFFER_SECONDS = 3  # Aumentado de 2 a 3 segundos para capturar más contexto
OVERLAP_SECONDS = 1.5  # Superposición entre segmentos de audio para mantener contexto

# Dispositivos: nombres o índices (ajustar al entorno del usuario)
MIC_DEVICE = 1       # Microfono - Focusrite USB (Focusrite USB Audio)
DISCORD_DEVICE = None   # Desactivamos temporalmente la captura de Discord para estabilidad

# Configuración de debug y transcripción
MIN_AUDIO_LEVEL = 0.015  # Aumentado de 0.01 a 0.015 para evitar ruido de fondo
SILENCE_SKIP = True    # Saltar audio silencioso para evitar errores
DEBUG_MODE = True      # Activa mensajes de depuración detallados
LANGUAGE = "es"        # Idioma para la transcripción: "es" para español
SHOW_TIMESTAMPS = True  # Mostrar marcas de tiempo en los mensajes
SILENCE_TIMEOUT = 2.0   # Segundos de silencio antes de considerar que el hablante terminó
MAX_REPETITIONS = 3    # Máximo número de repeticiones permitidas en una transcripción
DETECT_REPETITIONS = True  # Activar detección de repeticiones
USE_PREVIOUS_TEXT = True  # Usar texto anterior como contexto para mejorar coherencia
CONTEXT_SENTENCES = 2   # Número de oraciones anteriores para usar como contexto
USE_BEAM_SEARCH = True  # Usar búsqueda en haz para mejorar la calidad de transcripción
MODEL_SIZE = "base"     # Cambiar de "tiny" a "base" para mejor calidad (opciones: tiny, base, small, medium)
MIN_TEXT_LENGTH = 1     # Modificado: ahora acepta palabras individuales (antes 3)
CONFIDENCE_THRESHOLD = 0.4  # Reducido de 0.5 a 0.4 para ser menos estricto
FILTER_SHORT_PHRASES = False  # Nueva opción para desactivar el filtrado por longitud
HALLUCINATION_PATTERNS = [
    "¿eh?", "eh", "umm", "hmm", "uh", "ah", "oh", "este", "em", "mm"
]  # Patrones típicos de alucinaciones a filtrar
RESET_CONTEXT_AFTER_SILENCE = True  # Reiniciar contexto después de un silencio prolongado
MAX_SILENCE_BEFORE_RESET = 5.0  # Segundos de silencio antes de reiniciar el contexto

# Configuración de visualización
FONT_SIZE = 16  # Reducido de 20
CHAT_FONT_SIZE = 14  # Reducido de 18
BG_OPACITY = 150     # 0-255, donde 255 es completamente opaco
CHAT_HEIGHT = 200    # Reducido de 300 - Altura de la ventana de chat en píxeles
CHAT_WIDTH = 400     # Nuevo - Ancho de la ventana de chat en píxeles
MAX_CHAT_MESSAGES = 8  # Reducido de 15 - Máximo número de mensajes en el historial de chat
CHAT_MODE = True     # Activar el modo de chat vs. modo overlay tradicional
POSITION_BOTTOM_RIGHT = True  # Posicionar en la esquina inferior derecha en lugar de ocupar todo el ancho

# Colores para los diferentes hablantes
COLORS = {
    'mic': '#FF5555',   # Rojo para ti (micrófono)
    'discord': '#55AAFF',  # Azul para Discord
    'system': '#AAAAAA'  # Gris para mensajes del sistema
}

# Nombres de los hablantes
SPEAKERS = {
    'mic': 'Tú',
    'discord': 'Discord',
    'system': 'Sistema'
}

# Variables globales para control de hilos
running = True       # Controla si los hilos deben seguir ejecutándose

def setup_logging():
    """Configura el sistema de logs para registrar la actividad de la aplicación."""
    if not LOGS_ENABLED:
        return None
    
    # Crear directorio de logs si no existe
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)
    
    # Nombre del archivo de log con fecha
    today = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"{LOG_FILE}_{today}.log"
    log_path = os.path.join(LOG_FOLDER, log_filename)
    
    # Configurar el logger
    logger = logging.getLogger('whisper_overlay')
    logger.setLevel(LOG_LEVEL)
    
    # Manejador para archivo
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(LOG_LEVEL)
    
    # Formato para los logs
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Añadir también salida a consola si está habilitado
    if LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Limpiar logs antiguos si hay más de MAX_LOG_FILES
    clean_old_logs()
    
    # Log inicial
    logger.info(f"=== INICIO DE SESIÓN - Discord Whisper Overlay ===")
    logger.info(f"Configuración: Modelo={MODEL_SIZE}, Dispositivo={DEVICE}, GPU={USE_GPU}")
    
    return logger

def clean_old_logs():
    """Elimina logs antiguos si hay más del límite establecido."""
    if not os.path.exists(LOG_FOLDER):
        return
    
    log_files = [os.path.join(LOG_FOLDER, f) for f in os.listdir(LOG_FOLDER) 
                if f.startswith(LOG_FILE) and f.endswith('.log')]
    
    # Ordenar por fecha de modificación (más antiguo primero)
    log_files.sort(key=lambda x: os.path.getmtime(x))
    
    # Eliminar archivos antiguos si hay más del límite
    if len(log_files) > MAX_LOG_FILES:
        files_to_delete = log_files[:-MAX_LOG_FILES]  # Mantener los MAX_LOG_FILES más recientes
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Log antiguo eliminado: {file_path}")
            except Exception as e:
                print(f"Error al eliminar log antiguo {file_path}: {str(e)}")

def log_stats(logger):
    """Hilo que registra estadísticas periódicas de la aplicación."""
    global running
    
    if not logger or not LOG_PERFORMANCE:
        return
    
    logger.info("Iniciando registro de estadísticas periódicas")
    
    start_time = time.time()
    last_log_time = start_time
    
    while running:
        current_time = time.time()
        
        # Registrar estadísticas cada LOG_STATS_INTERVAL segundos
        if current_time - last_log_time >= LOG_STATS_INTERVAL:
            # Tiempo total de ejecución
            uptime = current_time - start_time
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            # Uso de memoria
            process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
            memory_usage = f"{process.memory_info().rss / (1024 * 1024):.1f} MB" if process else "No disponible"
            
            # Uso de CPU
            cpu_percent = f"{process.cpu_percent():.1f}%" if process else "No disponible"
            
            # Estadísticas de GPU si está disponible
            gpu_stats = ""
            if torch.cuda.is_available() and USE_GPU:
                try:
                    gpu_memory_allocated = f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB"
                    gpu_memory_reserved = f"{torch.cuda.memory_reserved() / (1024**3):.2f} GB"
                    gpu_stats = f", GPU: {gpu_memory_allocated} / {gpu_memory_reserved}"
                except:
                    gpu_stats = ", GPU: Error al obtener estadísticas"
            
            # Registrar estadísticas
            logger.info(f"ESTADÍSTICAS - Tiempo activo: {uptime_str}, RAM: {memory_usage}, CPU: {cpu_percent}{gpu_stats}")
            
            last_log_time = current_time
        
        # Esperar un poco para no consumir CPU
        time.sleep(5)
    
    logger.info("Finalizado registro de estadísticas periódicas")

class TranscriptionOverlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Ventana sin bordes, transparente y siempre encima
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Obtener ancho de pantalla y definir franja superior
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), 150)

        # Layout horizontal para dos etiquetas
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(20)

        # Label para micrófono (rojo)
        self.mic_label = QtWidgets.QLabel("Habla por el micrófono...", self)
        self.mic_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.mic_label.setStyleSheet(f"color: #FF5555; font-size: {FONT_SIZE}px; background-color: rgba(0, 0, 0, {BG_OPACITY});")

        # Label para Discord (azul)
        self.discord_label = QtWidgets.QLabel("Audio de Discord aparecerá aquí...", self)
        self.discord_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.discord_label.setStyleSheet(f"color: #55AAFF; font-size: {FONT_SIZE}px; background-color: rgba(0, 0, 0, {BG_OPACITY});")

        self.layout.addWidget(self.mic_label)
        self.layout.addWidget(self.discord_label)
        self.setLayout(self.layout)

    def update_text(self, source, text):
        if source == 'mic':
            self.mic_label.setText(text)
        elif source == 'discord':
            self.discord_label.setText(text)
    
    def mousePressEvent(self, event):
        # Permite hacer click para cerrar la ventana
        if event.button() == QtCore.Qt.LeftButton:
            print("Click detectado - cerrando aplicación")
            self.close()
            QtWidgets.QApplication.quit()

class ChatOverlay(QtWidgets.QWidget):
    # Señal personalizada para actualizar el chat desde otros hilos
    update_signal = QtCore.pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        self.messages = []  # Lista de mensajes en el chat
        self.last_speaker = None  # Último hablante para agrupar mensajes consecutivos
        self.last_message_time = {}  # Almacena la última vez que cada fuente habló
        
        # Conectar la señal a la función que actualiza el chat
        self.update_signal.connect(self._add_message_safe)
        
        self.init_ui()
        
    def init_ui(self):
        # Ventana sin bordes, transparente y siempre encima
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool |
            QtCore.Qt.WindowTransparentForInput  # Permite que los clics pasen a través de la ventana
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)  # No roba el foco al aparecer

        # Obtener ancho de pantalla y colocar en la parte inferior derecha si está activado
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        if POSITION_BOTTOM_RIGHT:
            self.setGeometry(screen.width() - CHAT_WIDTH - 20, screen.height() - CHAT_HEIGHT - 20, CHAT_WIDTH, CHAT_HEIGHT)
        else:
            self.setGeometry(0, screen.height() - CHAT_HEIGHT - 50, screen.width(), CHAT_HEIGHT)

        # Layout vertical para los mensajes
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        
        # Área de texto para el chat
        self.chat_area = QtWidgets.QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # Desactivar barra de desplazamiento
        self.chat_area.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba(0, 0, 0, {BG_OPACITY});
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px;
                font-size: {CHAT_FONT_SIZE}px;
            }}
        """)
        
        # Añadir instrucción inicial
        self.add_message("system", "Inicia una conversación hablando por el micrófono")
        
        self.layout.addWidget(self.chat_area)
        self.setLayout(self.layout)
        
    def _add_message_safe(self, source, text):
        """Método seguro para añadir mensajes (llamado desde el hilo principal)"""
        # Solo añadir mensaje si tiene contenido
        if not text or text.strip() == "":
            return
        
        current_time = time.time()
        
        # Verificar si es un nuevo mensaje o continuación basado en el tiempo transcurrido
        new_message = True
        if source in self.last_message_time:
            time_since_last = current_time - self.last_message_time[source]
            # Si ha pasado menos tiempo que el umbral de silencio, se considera continuación
            if time_since_last < SILENCE_TIMEOUT:
                new_message = False
        
        # Actualizar el tiempo del último mensaje para esta fuente
        self.last_message_time[source] = current_time
            
        # Si es un nuevo mensaje o si el hablante cambió
        if new_message or source != self.last_speaker:
            # Generar timestamp para nuevo mensaje
            if SHOW_TIMESTAMPS:
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                # Crear nuevo mensaje con timestamp
                self.messages.append({
                    'source': source,
                    'speaker': SPEAKERS.get(source, source),
                    'text': text,
                    'time': current_time,
                    'timestamp': timestamp
                })
            else:
                # Crear nuevo mensaje sin timestamp
                self.messages.append({
                    'source': source,
                    'speaker': SPEAKERS.get(source, source),
                    'text': text,
                    'time': current_time
                })
            self.last_speaker = source
        else:
            # Continuación del mensaje anterior del mismo hablante
            self.messages[-1]['text'] += f" {text}"
            
        # Limitar el número de mensajes para no sobrecargar
        if len(self.messages) > MAX_CHAT_MESSAGES:
            self.messages = self.messages[-MAX_CHAT_MESSAGES:]
            
        # Actualizar la visualización del chat
        self.update_chat_display()
    
    def add_message(self, source, text):
        """Emite la señal para añadir un mensaje de manera segura entre hilos"""
        self.update_signal.emit(source, text)
    
    def update_chat_display(self):
        """Actualiza el área de texto con los mensajes actuales"""
        self.chat_area.clear()
        
        # Crear y aplicar el formato HTML para el chat
        html = ""
        for msg in self.messages:
            color = COLORS.get(msg['source'], '#FFFFFF')
            speaker = msg['speaker']
            text = msg['text']
            
            # Incluir timestamp si está disponible y activado
            if SHOW_TIMESTAMPS and 'timestamp' in msg:
                timestamp = msg['timestamp']
                # Formato HTML para el mensaje con timestamp
                html += f"""
                <div style='margin-bottom: 8px;'>
                    <span style='color: #999999; font-size: 14px;'>[{timestamp}]</span>
                    <span style='color: {color}; font-weight: bold;'> {speaker}:</span>
                    <span style='color: white;'> {text}</span>
                </div>
                """
            else:
                # Formato HTML para el mensaje sin timestamp
                html += f"""
                <div style='margin-bottom: 8px;'>
                    <span style='color: {color}; font-weight: bold;'>{speaker}:</span>
                    <span style='color: white;'> {text}</span>
                </div>
                """
        
        # Establecer el HTML en el área de texto
        self.chat_area.setHtml(html)
        
        # Desplazar al final para ver los mensajes más recientes
        cursor = self.chat_area.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.chat_area.setTextCursor(cursor)
    
    def update_text(self, source, text):
        """Método compatible con TranscriptionOverlay para recibir actualizaciones"""
        self.add_message(source, text)
        
    def mousePressEvent(self, event):
        # Permite hacer click para cerrar la ventana
        if event.button() == QtCore.Qt.LeftButton:
            print("Click detectado - cerrando aplicación")
            self.close()
            QtWidgets.QApplication.quit()
            
    def mouseMoveEvent(self, event):
        # Permite arrastrar la ventana
        if event.buttons() & QtCore.Qt.LeftButton:
            self.move(self.mapToGlobal(event.pos() - QtCore.QPoint(self.width() // 2, self.height() // 2)))
            event.accept()

def audio_capture_mic(device_index, frames_queue):
    """Lee datos del micrófono y los almacena en la cola."""
    global running
    p = pyaudio.PyAudio()
    
    # Si el dispositivo es None, usar el dispositivo de entrada predeterminado
    if device_index is None:
        try:
            device_index = p.get_default_input_device_info()['index']
            print(f"Usando dispositivo de entrada predeterminado (índice {device_index})")
        except Exception as e:
            print(f"Error al obtener dispositivo predeterminado: {str(e)}")
            device_index = 0  # Intentar con el primer dispositivo
    
    stream = None
    try:
        print(f"Intentando abrir stream de micrófono en dispositivo {device_index}...")
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        print(f"Captura de audio de micrófono iniciada para dispositivo {device_index}")
        
        while running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames_queue.append(data)
            except Exception as e:
                print(f"Error durante la captura de audio del micrófono: {str(e)}")
                time.sleep(0.5)  # Esperar un poco antes de intentar nuevamente
    except Exception as e:
        print(f"Error crítico al abrir stream para dispositivo de micrófono {device_index}: {str(e)}")
        print(f"Detalles del error: {traceback.format_exc()}")
        # Notificar al usuario en la terminal
        print("\nERROR: No se pudo abrir el dispositivo de audio para el micrófono.")
        print("Posibles soluciones:")
        print("1. Verifica que el dispositivo de micrófono esté conectado y funcionando")
        print("2. Ejecuta device_list.py para ver los dispositivos disponibles")
        print("3. Modifica MIC_DEVICE en el código con un índice correcto\n")
        
        # En caso de error, simular audio silencioso para evitar que el hilo se detenga
        print("Usando audio simulado para el micrófono...")
        while running:
            silent_frame = b'\x00' * CHUNK * CHANNELS * 2  # 2 bytes por muestra (FORMAT=paInt16)
            frames_queue.append(silent_frame)
            time.sleep(0.05)
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
        try:
            p.terminate()
        except:
            pass
        print("Hilo de captura de micrófono finalizado")

def simulate_discord_audio(frames_queue):
    """Genera frames de audio silencioso para Discord."""
    global running
    print("Usando audio simulado para Discord...")
    while running:
        silent_frame = b'\x00' * CHUNK * CHANNELS * 2
        frames_queue.append(silent_frame)
        time.sleep(0.05)

def simulate_discord_conversation(overlay):
    """Simula respuestas de Discord para propósitos de demostración"""
    global running
    
    # Lista de posibles respuestas para la simulación
    responses = [
        "Hola, ¿cómo estás hoy?",
        "Entiendo lo que dices",
        "¿Podemos hablar sobre el proyecto?",
        "Esta transcripción funciona muy bien",
        "Me encanta este modo de chat",
        "¿Puedes repetir eso por favor?",
        "¿Te gustaría jugar?",
        "La reunión empieza en 5 minutos",
        "Necesito tu ayuda con algo",
        "¿Has visto el nuevo video?"
    ]
    
    # Esperar un tiempo inicial para dar tiempo a hablar primero
    time.sleep(10)
    
    # Mientras se ejecuta el programa, simular respuestas de Discord
    while running:
        # Elegir una respuesta aleatoria
        if random.random() < 0.3:  # 30% de probabilidad de responder
            response = random.choice(responses)
            overlay.update_text('discord', response)
            print(f"[discord] Simulación: {response}")
            
        # Esperar entre 10 y 20 segundos para la próxima respuesta
        time.sleep(random.uniform(10, 20))

def transcribe_loop(source, model, frames_queue, overlay, logger=None):
    """Procesa la cola cada X segundos, envía a Whisper y actualiza overlay."""
    global running
    buffer_size = int(RATE / CHUNK * BUFFER_SECONDS)
    overlap_size = int(RATE / CHUNK * OVERLAP_SECONDS)
    last_error_time = 0
    error_count = 0
    previous_text = ""
    last_silence_time = time.time()
    silence_detected = False
    total_transcriptions = 0
    silence_periods = 0
    last_stats_log = time.time()
    
    log_prefix = f"[{source.upper()}]"
    
    if logger:
        logger.info(f"{log_prefix} Iniciando bucle de transcripción con modelo {MODEL_SIZE}")
    
    print(f"Iniciando bucle de transcripción para {source}")
    
    while running:
        try:
            current_time = time.time()
            
            # Registrar estadísticas periódicas de este hilo de transcripción
            if logger and LOG_PERFORMANCE and (current_time - last_stats_log >= LOG_STATS_INTERVAL):
                logger.info(f"{log_prefix} Estadísticas - Transcripciones: {total_transcriptions}, Periodos de silencio: {silence_periods}")
                last_stats_log = current_time
            
            if len(frames_queue) >= buffer_size:
                # Obtener datos y limpiar cola
                frames = [frames_queue.popleft() for _ in range(buffer_size)]
                audio_bytes = b"".join(frames)
                
                # Convertir a float32 numpy
                audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
                
                # Verificar nivel de audio (evita errores con silencio)
                audio_level = np.max(np.abs(audio_np))
                
                if SILENCE_SKIP and audio_level < MIN_AUDIO_LEVEL:
                    if DEBUG_MODE and source == 'mic':  # Solo para el micrófono para no llenar la consola
                        print(f"[{source}] Audio silencioso detectado (nivel: {audio_level:.4f})")
                    
                    # Verificar si el silencio es prolongado para reiniciar contexto
                    if not silence_detected:
                        silence_detected = True
                        silence_periods += 1
                        last_silence_time = current_time
                        if logger and LOG_LEVEL <= logging.DEBUG:
                            logger.debug(f"{log_prefix} Silencio detectado (nivel: {audio_level:.4f})")
                    elif RESET_CONTEXT_AFTER_SILENCE and (current_time - last_silence_time > MAX_SILENCE_BEFORE_RESET):
                        if previous_text and DEBUG_MODE:
                            print(f"[{source}] Silencio prolongado detectado, reiniciando contexto")
                            if logger:
                                logger.info(f"{log_prefix} Silencio prolongado ({current_time - last_silence_time:.1f}s), reiniciando contexto")
                        previous_text = ""  # Reiniciar contexto después de silencio prolongado
                    
                    time.sleep(0.1)
                    continue
                else:
                    # Reiniciar flag de silencio cuando se detecta audio
                    silence_detected = False
                
                # Verificar que el audio no sea completamente ceros o tenga una forma incorrecta
                if len(audio_np) == 0 or np.all(audio_np == 0):
                    if DEBUG_MODE and source == 'mic':
                        print(f"[{source}] Audio vacío detectado, saltando")
                    time.sleep(0.1)
                    continue
                
                try:
                    # Transcribir con idioma español
                    transcription_start = time.time()
                    result = model.transcribe(
                        audio_np, 
                        fp16=HALF_PRECISION if DEVICE == "cuda" else False, 
                        language=LANGUAGE,  # Usar español como idioma
                        task="transcribe",   # Tarea de transcripción
                        beam_size=5 if USE_BEAM_SEARCH else None,  # Usar búsqueda en haz si está activado
                        initial_prompt=previous_text if USE_PREVIOUS_TEXT and previous_text else None  # Usar texto anterior como contexto
                    )
                    transcription_time = time.time() - transcription_start
                    text = result['text'].strip()
                    
                    # Detectar y eliminar repeticiones excesivas si está activado
                    if DETECT_REPETITIONS:
                        # 1. Filtro básico de repeticiones de palabras
                        words = text.split()
                        filtered_words = []
                        repetition_count = 0
                        last_word = None
                        
                        for word in words:
                            if word == last_word:
                                repetition_count += 1
                            else:
                                repetition_count = 0
                            
                            if repetition_count < MAX_REPETITIONS:
                                filtered_words.append(word)
                            
                            last_word = word
                        
                        text = " ".join(filtered_words)
                        
                        # 2. Filtro avanzado para patrones repetitivos "¿eh?" y similares
                        for pattern in HALLUCINATION_PATTERNS:
                            # Contar ocurrencias
                            pattern_count = text.lower().count(pattern)
                            
                            # Si hay más de MAX_REPETITIONS ocurrencias, filtrar todas
                            if pattern_count > MAX_REPETITIONS:
                                # Reemplazar el patrón con una sola ocurrencia
                                text = re.sub(f"(?i){re.escape(pattern)}\\s*", f"{pattern} ", text, count=1)
                                # Eliminar las ocurrencias restantes
                                text = re.sub(f"(?i){re.escape(pattern)}\\s*", "", text)
                    
                    # 3. Filtrar expresiones cortas de la lista de alucinaciones
                    words = text.split()
                    filtered_words = []
                    for word in words:
                        # Filtrar palabras completas que coincidan con patrones de alucinación
                        if not FILTER_SHORT_PHRASES or word.lower() not in HALLUCINATION_PATTERNS:
                            filtered_words.append(word)
                    
                    text = " ".join(filtered_words)
                    
                    # 4. Verificar si el texto debe ignorarse por ser muy corto
                    # Pero permitir palabras cortas importantes como "Hola", "Sí", etc.
                    is_too_short = False
                    if FILTER_SHORT_PHRASES:
                        word_count = len(text.split())
                        # Ignorar textos muy cortos que probablemente sean ruido
                        if word_count < MIN_TEXT_LENGTH:
                            # Solo descartar si es una palabra que está en el patrón de alucinaciones
                            if text.lower() in HALLUCINATION_PATTERNS:
                                is_too_short = True
                                if DEBUG_MODE:
                                    print(f"[{source}] Texto demasiado corto, ignorado: {text}")
                    
                    if is_too_short:
                        time.sleep(0.1)
                        continue
                    
                    # 5. Filtrar transcripciones con baja confianza
                    # Obtener la confianza de manera segura (compatible con CPU y CUDA)
                    try:
                        # Formato antiguo (CPU)
                        avg_confidence = sum(segment.avg_logprob for segment in result["segments"]) / len(result["segments"]) if result["segments"] else -1
                    except AttributeError:
                        try:
                            # Formato nuevo (CUDA)
                            avg_confidence = sum(segment.get('avg_logprob', -1) for segment in result["segments"]) / len(result["segments"]) if result["segments"] else -1
                        except:
                            # Si falla cualquier método, usar un valor predeterminado
                            if DEBUG_MODE:
                                print(f"[{source}] No se pudo determinar la confianza, usando valor predeterminado")
                            avg_confidence = -1
                    
                    normalized_confidence = np.exp(avg_confidence) if avg_confidence > -float('inf') else 0  # Convertir log-prob a probabilidad, con protección
                    
                    # Registrar información sobre alucinaciones detectadas para análisis
                    text_before_filtering = result['text'].strip()
                    if logger and LOG_LEVEL <= logging.DEBUG and text_before_filtering != text:
                        # Registrar cuando se han filtrado alucinaciones o repeticiones
                        logger.debug(f"{log_prefix} Texto original: '{text_before_filtering}' → Texto filtrado: '{text}'")
                    
                    # Reducir el umbral para palabras cortas (para permitir que "Hola" pase)
                    adjusted_threshold = CONFIDENCE_THRESHOLD
                    if len(text.split()) <= 2:  # Para palabras o frases muy cortas
                        adjusted_threshold = CONFIDENCE_THRESHOLD * 0.8  # Reducir el umbral en 20%
                    
                    if normalized_confidence < adjusted_threshold:
                        if DEBUG_MODE:
                            print(f"[{source}] Baja confianza ({normalized_confidence:.4f}), ignorado: {text}")
                        time.sleep(0.1)
                        continue
                    
                    # Solo actualizar si hay texto después de todos los filtros
                    if text:
                        overlay.update_text(source, text)
                        print(f"[{source}] Transcripción: {text}")
                        total_transcriptions += 1
                        
                        # Logging de la transcripción si está habilitado
                        if logger and LOG_TRANSCRIPTIONS:
                            timestamp = time.strftime("%H:%M:%S", time.localtime())
                            confidence_str = f", confianza: {normalized_confidence:.2f}" if normalized_confidence > 0 else ""
                            perf_str = f", tiempo: {transcription_time:.2f}s" if LOG_PERFORMANCE else ""
                            logger.info(f"{log_prefix} [{timestamp}] {SPEAKERS.get(source, source)}: {text}{confidence_str}{perf_str}")
                        
                        # Actualizar texto anterior para contexto, pero limitar a las últimas CONTEXT_SENTENCES
                        if USE_PREVIOUS_TEXT:
                            # Dividir en oraciones (aproximado)
                            sentences = re.split(r'[.!?]+', previous_text)
                            # Quedarse con las últimas CONTEXT_SENTENCES oraciones significativas
                            significant_sentences = [s for s in sentences if len(s.strip()) > 3][-CONTEXT_SENTENCES:]
                            # Reconstruir contexto
                            previous_text = ". ".join(significant_sentences) + ". " + text
                        
                        # Resetear contador de errores si hay transcripción exitosa
                        error_count = 0
                except Exception as e:
                    current_time = time.time()
                    error_count += 1
                    
                    # Limitar los mensajes de error para no saturar la consola
                    if current_time - last_error_time > 5:  # máximo un mensaje cada 5 segundos
                        error_msg = f"[{source}] Error en la transcripción: {str(e)}"
                        print(error_msg)
                        if logger:
                            logger.error(f"{log_prefix} {str(e)}")
                        last_error_time = current_time
                    
                    # Si hay muchos errores consecutivos, esperar más tiempo
                    if error_count > 10:
                        critical_msg = f"[{source}] Demasiados errores consecutivos, esperando más tiempo..."
                        print(critical_msg)
                        if logger:
                            logger.warning(f"{log_prefix} Detectados {error_count} errores consecutivos")
                        time.sleep(2)
                        error_count = 0
                    else:
                        time.sleep(0.5)
            else:
                # Esperar a que haya suficientes frames
                time.sleep(0.1)
        except Exception as e:
            error_msg = f"[{source}] Error en el bucle de transcripción: {str(e)}"
            print(error_msg)
            if logger:
                logger.error(f"{log_prefix} Error general: {str(e)}")
                if LOG_LEVEL <= logging.DEBUG:
                    logger.debug(f"{log_prefix} Traza: {traceback.format_exc()}")
            time.sleep(1)  # Esperar más tiempo en caso de error general
    
    # Mensaje de finalización
    end_msg = f"Bucle de transcripción para {source} finalizado"
    print(end_msg)
    if logger:
        logger.info(f"{log_prefix} Finalizado. Total transcripciones: {total_transcriptions}")

def main():
    global running
    running = True
    
    try:
        print("Iniciando Discord Whisper Overlay - Versión Estable...")
        
        # Configurar sistema de logs
        logger = setup_logging()
        if logger:
            logger.info("Iniciando Discord Whisper Overlay")
            
        # Mostrar información de hardware y procesamiento
        if torch.cuda.is_available() and USE_GPU and not FORCE_CPU:
            gpu_info = f"GPU detectada: {torch.cuda.get_device_name(0)}"
            compute_capability = torch.cuda.get_device_capability(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convertir a GB
            
            print("\n" + "=" * 60)
            print(f"USANDO ACELERACIÓN POR GPU (CUDA)")
            print(f"- {gpu_info}")
            print(f"- Capacidad de cómputo: {compute_capability[0]}.{compute_capability[1]}")
            print(f"- Memoria VRAM: {vram:.2f} GB")
            print(f"- Precisión: {'Media (FP16)' if HALF_PRECISION else 'Completa (FP32)'}")
            print("=" * 60 + "\n")
        else:
            reason = "Forzado por configuración" if FORCE_CPU else "No se detectó GPU compatible"
            if not torch.cuda.is_available():
                reason = "CUDA no disponible - Verifica la instalación de PyTorch con CUDA"
            
            print("\n" + "=" * 60)
            print(f"USANDO PROCESAMIENTO POR CPU ({reason})")
            print(f"- Modelo Whisper: {MODEL_SIZE}")
            print(f"- El procesamiento será más lento que con GPU")
            print("=" * 60 + "\n")
        
        # Cargar modelo Whisper (tiny/base/small según poder)
        print("Cargando modelo Whisper (esto puede tardar unos segundos)...")
        model = whisper.load_model(MODEL_SIZE, device=DEVICE)  # Modelo ajustado según configuración
        print(f"Modelo '{MODEL_SIZE}' cargado correctamente en {DEVICE}")

        # Crear aplicación y overlay
        print("Creando interfaz de usuario...")
        app = QtWidgets.QApplication(sys.argv)
        overlay = ChatOverlay() if CHAT_MODE else TranscriptionOverlay()
        overlay.show()
        print("Interfaz creada correctamente")

        # Colas para audio
        mic_queue = deque(maxlen=1000)  # Limitar tamaño de la cola para evitar fuga de memoria
        discord_queue = deque(maxlen=1000)
        
        print(f"\nIniciando captura de audio con configuración:")
        print(f"MIC_DEVICE: {MIC_DEVICE}")
        print(f"DISCORD_DEVICE: {DISCORD_DEVICE}")
        print(f"BUFFER_SECONDS: {BUFFER_SECONDS}")
        print(f"DEBUG_MODE: {DEBUG_MODE}")

        # Hilos de captura
        try:
            # Hilo para el micrófono
            mic_thread = threading.Thread(
                target=audio_capture_mic,
                args=(MIC_DEVICE, mic_queue),
                daemon=True,
                name="MicThread"
            )
            
            # Para Discord usamos simulación por ahora para mayor estabilidad
            discord_thread = threading.Thread(
                target=simulate_discord_audio,
                args=(discord_queue,),
                daemon=True,
                name="DiscordThread"
            )
            
            mic_thread.start()
            print("Hilo de captura de micrófono iniciado")
            
            discord_thread.start()
            print("Hilo de audio de Discord iniciado (simulado)")
        except Exception as e:
            print(f"Error al iniciar hilos de captura de audio: {str(e)}")
            traceback.print_exc()
            time.sleep(10)
            raise

        # Hilos de transcripción
        try:
            mic_transcribe = threading.Thread(
                target=transcribe_loop,
                args=('mic', model, mic_queue, overlay, logger),
                daemon=True,
                name="MicTranscribeThread"
            )
            
            discord_transcribe = threading.Thread(
                target=transcribe_loop,
                args=('discord', model, discord_queue, overlay, logger),
                daemon=True,
                name="DiscordTranscribeThread"
            )
            
            mic_transcribe.start()
            print("Hilo de transcripción de micrófono iniciado")
            
            discord_transcribe.start()
            print("Hilo de transcripción de Discord iniciado")
        except Exception as e:
            print(f"Error al iniciar hilos de transcripción: {str(e)}")
            traceback.print_exc()
            time.sleep(10)
            raise

        # Hilo para simular conversación de Discord
        try:
            discord_simulation_thread = threading.Thread(
                target=simulate_discord_conversation,
                args=(overlay,),
                daemon=True,
                name="DiscordSimulationThread"
            )
            discord_simulation_thread.start()
            print("Hilo de simulación de conversación de Discord iniciado")
        except Exception as e:
            print(f"Error al iniciar hilo de simulación de conversación de Discord: {str(e)}")
            traceback.print_exc()
            time.sleep(10)
            raise

        # Hilo para registrar estadísticas periódicas
        try:
            stats_thread = threading.Thread(
                target=log_stats,
                args=(logger,),
                daemon=True,
                name="StatsThread"
            )
            stats_thread.start()
            print("Hilo de registro de estadísticas iniciado")
        except Exception as e:
            print(f"Error al iniciar hilo de estadísticas: {str(e)}")
            traceback.print_exc()
            time.sleep(10)
            raise

        print("\nDiscord Whisper Overlay iniciado correctamente")
        print("Habla por el micrófono para ver tu transcripción (en rojo)")
        print("El audio de Discord aparecería en azul (actualmente simulado)")
        print("Haz click en la ventana para cerrar la aplicación")
        
        # Conectar señal de cierre de aplicación
        app.aboutToQuit.connect(lambda: setattr(sys.modules[__name__], 'running', False))
        
        # Registrar inicio completo del sistema en el log
        if logger:
            logger.info("Sistema iniciado completamente. Aplicación operativa.")
            if not PSUTIL_AVAILABLE:
                logger.warning("Módulo psutil no encontrado. Las estadísticas de rendimiento serán limitadas.")
        
        return app.exec_()
    except Exception as e:
        running = False
        error_msg = f"ERROR CRÍTICO: {str(e)}"
        print(error_msg)
        if logger:
            logger.critical(f"Error crítico en la aplicación: {str(e)}")
            logger.critical(f"Traza: {traceback.format_exc()}")
        traceback.print_exc()
        print("\nEl programa se cerrará en 30 segundos...")
        time.sleep(30)
        return 1
    finally:
        # Asegurarse de que la variable running se establezca a False al salir
        running = False
        
        # Registrar finalización de la aplicación
        if 'logger' in locals() and logger:
            logger.info("=== FIN DE SESIÓN - Discord Whisper Overlay ===")

if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"Programa finalizado con código: {exit_code}")
    except Exception as e:
        print(f"Error no capturado: {str(e)}")
        traceback.print_exc()
    finally:
        print("Presiona Enter para cerrar esta ventana...")
        input()
        sys.exit(0)
`

# discord_whisper_overlay.py

import sys
import threading
import time
from collections import deque

import numpy as np
import pyaudio
import whisper
from PyQt5 import QtWidgets, QtCore, QtGui

# CONFIGURACIÓN DE AUDIO
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
BUFFER_SECONDS = 2  # buffer reducido para menor latencia

# Dispositivos: nombres o índices (ajustar al entorno del usuario)
MIC_DEVICE = 1       # Microfono - Focusrite USB (Focusrite USB Audio)
DISCORD_DEVICE = None   # Configurado como None para esta versión

# Configuración de debug y transcripción
DEBUG_MODE = True    # Poner True para ver mensajes de debug detallados
MIN_AUDIO_LEVEL = 0.01  # Nivel mínimo de audio para transcribir (evita silencio)
SILENCE_SKIP = True    # Saltar audio silencioso para evitar errores

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

        # Obtener ancho de pantalla y definir franja superior de 100px de alto
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), 100)

        # Layout horizontal para dos etiquetas
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(20)

        # Label para micrófono (rojo)
        self.mic_label = QtWidgets.QLabel("", self)
        self.mic_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.mic_label.setStyleSheet("color: #FF5555; font-size: 24px;")

        # Label para Discord (azul)
        self.discord_label = QtWidgets.QLabel("", self)
        self.discord_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.discord_label.setStyleSheet("color: #55AAFF; font-size: 24px;")

        self.layout.addWidget(self.mic_label)
        self.layout.addWidget(self.discord_label)
        self.setLayout(self.layout)
        self.show()

    def update_text(self, source: str, new_text: str):
        # Actualiza el label correspondiente
        if source == 'mic':
            self.mic_label.setText(new_text)
        elif source == 'discord':
            self.discord_label.setText(new_text)


def audio_capture(device_index, frames_queue):
    """Lee datos de un dispositivo y los almacena en la cola."""
    p = pyaudio.PyAudio()
    
    # Si el dispositivo es None, usar el dispositivo de entrada predeterminado
    if device_index is None:
        device_index = p.get_default_input_device_info()['index']
        print(f"Usando dispositivo de entrada predeterminado (índice {device_index})")
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        print(f"Captura de audio iniciada para dispositivo {device_index}")
        
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames_queue.append(data)
            except Exception as e:
                print(f"Error durante la captura de audio: {str(e)}")
                time.sleep(0.5)  # Esperar un poco antes de intentar nuevamente
    except Exception as e:
        print(f"Error al abrir stream para dispositivo {device_index}: {str(e)}")
        # En caso de error, simular audio silencioso para evitar que el hilo se detenga
        while True:
            silent_frame = b'\x00' * CHUNK * CHANNELS * 2  # 2 bytes por muestra (FORMAT=paInt16)
            frames_queue.append(silent_frame)
            time.sleep(0.05)


def transcribe_loop(source: str, model, frames_queue, overlay):
    """Procesa la cola cada X segundos, envía a Whisper y actualiza overlay."""
    buffer_size = int(RATE / CHUNK * BUFFER_SECONDS)
    while True:
        try:
            if len(frames_queue) >= buffer_size:
                # Obtener datos y limpiar cola
                frames = [frames_queue.popleft() for _ in range(buffer_size)]
                audio_bytes = b"".join(frames)
                
                # Convertir a float32 numpy
                audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
                
                # Verificar nivel de audio (evita errores con silencio)
                audio_level = np.max(np.abs(audio_np))
                if SILENCE_SKIP and audio_level < MIN_AUDIO_LEVEL:
                    if DEBUG_MODE:
                        print(f"[{source}] Audio silencioso detectado (nivel: {audio_level:.4f})")
                    time.sleep(0.1)
                    continue
                
                # Verificar que el audio no sea completamente ceros o tenga una forma incorrecta
                if len(audio_np) == 0 or np.all(audio_np == 0):
                    if DEBUG_MODE:
                        print(f"[{source}] Audio vacío detectado, saltando")
                    time.sleep(0.1)
                    continue
                
                try:
                    # Transcribir
                    result = model.transcribe(audio_np, fp16=False)
                    text = result['text'].strip()
                    
                    # Solo actualizar si hay texto
                    if text:
                        overlay.update_text(source, text)
                        if DEBUG_MODE:
                            print(f"[{source}] Transcripción: {text}")
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"[{source}] Error en la transcripción: {str(e)}")
                    time.sleep(0.5)  # Esperar antes de intentar nuevamente
            else:
                # Esperar a que haya suficientes frames
                time.sleep(0.1)
        except Exception as e:
            print(f"[{source}] Error en el bucle de transcripción: {str(e)}")
            time.sleep(1)  # Esperar más tiempo en caso de error general


def main():
    try:
        print("Iniciando Discord Whisper Overlay...")
        
        # Cargar modelo Whisper (tiny/base/small según poder)
        print("\nCargando modelo Whisper (esto puede tardar unos segundos)...")
        model = whisper.load_model("tiny")  # Cambiado a "tiny" para más velocidad y menos errores
        print("Modelo cargado correctamente")

        # Crear aplicación y overlay
        print("Creando interfaz de usuario...")
        app = QtWidgets.QApplication(sys.argv)
        overlay = TranscriptionOverlay()
        print("Interfaz creada correctamente")

        # Colas para mic y Discord
        mic_queue = deque()
        discord_queue = deque()
        
        print(f"\nIniciando captura de audio con configuración:")
        print(f"MIC_DEVICE: {MIC_DEVICE}")
        print(f"DISCORD_DEVICE: {DISCORD_DEVICE}")
        print(f"BUFFER_SECONDS: {BUFFER_SECONDS}")
        print(f"DEBUG_MODE: {DEBUG_MODE}")

        # Hilos de captura
        try:
            mic_thread = threading.Thread(
                target=audio_capture,
                args=(MIC_DEVICE, mic_queue),
                daemon=True
            )
            
            discord_thread = threading.Thread(
                target=audio_capture,
                args=(DISCORD_DEVICE, discord_queue),
                daemon=True
            )
            
            mic_thread.start()
            print("Hilo de captura de micrófono iniciado")
            
            discord_thread.start()
            print("Hilo de captura de Discord iniciado")
        except Exception as e:
            print(f"Error al iniciar hilos de captura de audio: {str(e)}")
            import traceback
            traceback.print_exc()
            time.sleep(10)  # Esperar 10 segundos para poder leer el error
            raise

        # Hilos de transcripción
        try:
            mic_transcribe = threading.Thread(
                target=transcribe_loop,
                args=('mic', model, mic_queue, overlay),
                daemon=True
            )
            
            discord_transcribe = threading.Thread(
                target=transcribe_loop,
                args=('discord', model, discord_queue, overlay),
                daemon=True
            )
            
            mic_transcribe.start()
            print("Hilo de transcripción de micrófono iniciado")
            
            discord_transcribe.start()
            print("Hilo de transcripción de Discord iniciado")
        except Exception as e:
            print(f"Error al iniciar hilos de transcripción: {str(e)}")
            import traceback
            traceback.print_exc()
            time.sleep(10)  # Esperar 10 segundos para poder leer el error
            raise

        print("\nDiscord Whisper Overlay iniciado correctamente")
        print("Habla por el micrófono o reproduce audio en Discord para ver la transcripción")
        
        # Asegurarnos de que la interfaz permanezca visible
        overlay.show()
        
        # Ejecutar la aplicación
        return app.exec_()
    except Exception as e:
        print(f"ERROR CRÍTICO: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nEl programa se cerrará en 30 segundos...")
        time.sleep(30)  # Esperar 30 segundos para poder leer el error
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(f"Programa finalizado con código: {exit_code}")
    input("Presiona Enter para cerrar esta ventana...")  # Esperar input antes de cerrar
    sys.exit(exit_code)

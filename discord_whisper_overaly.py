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
MIC_DEVICE = None       # None usa dispositivo por defecto
DISCORD_DEVICE = None   # especificar nombre o índice de cable virtual

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
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames_queue.append(data)


def transcribe_loop(source: str, model, frames_queue, overlay):
    """Procesa la cola cada X segundos, envía a Whisper y actualiza overlay."""
    buffer_size = int(RATE / CHUNK * BUFFER_SECONDS)
    while True:
        if len(frames_queue) >= buffer_size:
            # Obtener datos y limpiar cola
            frames = [frames_queue.popleft() for _ in range(buffer_size)]
            audio_bytes = b"".join(frames)
            # Convertir a float32 numpy
            audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
            # Transcribir
            result = model.transcribe(audio_np, fp16=False)
            overlay.update_text(source, result['text'])
        time.sleep(0.1)


def main():
    # Cargar modelo Whisper (tiny/base/small según poder)
    model = whisper.load_model("small")

    # Crear aplicación y overlay
    app = QtWidgets.QApplication(sys.argv)
    overlay = TranscriptionOverlay()

    # Colas para mic y Discord
    mic_queue = deque()
    discord_queue = deque()

    # Hilos de captura
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
    discord_thread.start()

    # Hilos de transcripción
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
    discord_transcribe.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

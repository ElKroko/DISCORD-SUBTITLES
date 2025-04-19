import sys
import threading
import time
import random
from PyQt5 import QtWidgets, QtCore, QtGui

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
        self.mic_label = QtWidgets.QLabel("Transcripción del micrófono aparecerá aquí", self)
        self.mic_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.mic_label.setStyleSheet("color: #FF5555; font-size: 24px; background-color: rgba(0, 0, 0, 100);")

        # Label para Discord (azul)
        self.discord_label = QtWidgets.QLabel("Transcripción de Discord aparecerá aquí", self)
        self.discord_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.discord_label.setStyleSheet("color: #55AAFF; font-size: 24px; background-color: rgba(0, 0, 0, 100);")

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

# Frases de ejemplo para simular transcripciones
MIC_PHRASES = [
    "Hola, estoy hablando por el micrófono",
    "Esta es una prueba de transcripción",
    "El overlay está funcionando correctamente",
    "Esto simula lo que dirías por el micrófono",
    "La transcripción funciona en tiempo real"
]

DISCORD_PHRASES = [
    "Hola, soy tu amigo en Discord",
    "Te estoy hablando desde el chat de voz",
    "¿Me escuchas correctamente?",
    "Esta es una simulación de la transcripción de Discord",
    "El texto en azul representa lo que otros dicen"
]

def simulate_transcription(source, phrases, overlay):
    """Simula transcripciones sin usar Whisper ni audio real"""
    while True:
        try:
            # Seleccionar una frase aleatoria
            text = random.choice(phrases)
            
            # Actualizar el overlay con la frase
            overlay.update_text(source, text)
            print(f"[{source}] Transcripción simulada: {text}")
            
            # Esperar entre 3 y 5 segundos para la próxima frase
            time.sleep(random.uniform(3, 5))
        except Exception as e:
            print(f"Error en la simulación de {source}: {str(e)}")
            time.sleep(1)

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        
        print("Creando Discord Whisper Overlay (versión simulada)...")
        overlay = TranscriptionOverlay()
        overlay.show()
        
        print("Iniciando hilos de simulación de transcripción...")
        
        # Iniciar hilos para simular transcripciones
        mic_thread = threading.Thread(
            target=simulate_transcription,
            args=('mic', MIC_PHRASES, overlay),
            daemon=True
        )
        
        discord_thread = threading.Thread(
            target=simulate_transcription,
            args=('discord', DISCORD_PHRASES, overlay),
            daemon=True
        )
        
        mic_thread.start()
        discord_thread.start()
        
        print("Overlay iniciado con transcripciones simuladas")
        print("Haz click en la ventana para cerrar la aplicación")
        
        return app.exec_()
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        time.sleep(10)
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"Programa finalizado con código: {exit_code}")
    input("Presiona Enter para cerrar esta ventana...")
    sys.exit(exit_code)
import sys
import time
from PyQt5 import QtWidgets, QtCore, QtGui

class SimpleOverlay(QtWidgets.QWidget):
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
        self.mic_label = QtWidgets.QLabel("Este texto debería aparecer en ROJO", self)
        self.mic_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.mic_label.setStyleSheet("color: #FF5555; font-size: 24px; background-color: rgba(0, 0, 0, 100);")

        # Label para Discord (azul)
        self.discord_label = QtWidgets.QLabel("Este texto debería aparecer en AZUL", self)
        self.discord_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.discord_label.setStyleSheet("color: #55AAFF; font-size: 24px; background-color: rgba(0, 0, 0, 100);")

        self.layout.addWidget(self.mic_label)
        self.layout.addWidget(self.discord_label)
        self.setLayout(self.layout)

    def mousePressEvent(self, event):
        # Permite hacer click para cerrar la ventana
        if event.button() == QtCore.Qt.LeftButton:
            print("Click detectado - cerrando aplicación")
            self.close()
            QtWidgets.QApplication.quit()

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        
        print("Creando ventana overlay simple...")
        overlay = SimpleOverlay()
        overlay.show()
        
        print("Ventana creada. Haz click en la ventana para cerrarla.")
        print("¿Puedes ver el texto rojo y azul en la parte superior de la pantalla?")
        
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
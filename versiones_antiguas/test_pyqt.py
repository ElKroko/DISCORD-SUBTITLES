import sys
from PyQt5 import QtWidgets, QtCore

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Crear una ventana simple
    window = QtWidgets.QWidget()
    window.setWindowTitle("Test PyQt5")
    window.setGeometry(100, 100, 400, 200)
    
    # Añadir un texto
    label = QtWidgets.QLabel("Si puedes ver este mensaje, PyQt5 está funcionando correctamente", window)
    label.setGeometry(50, 50, 300, 100)
    
    # Mostrar la ventana
    window.show()
    
    print("Ventana de prueba creada. ¿Puedes verla?")
    
    # Ejecutar la aplicación
    return app.exec_()

if __name__ == "__main__":
    exit_code = main()
    print(f"Programa finalizado con código: {exit_code}")
    input("Presiona Enter para cerrar esta ventana...")
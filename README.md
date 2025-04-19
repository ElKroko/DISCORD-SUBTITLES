# Discord Whisper Overlay

Un overlay (capa superpuesta) para transcripción en tiempo real de audio del micrófono y Discord utilizando OpenAI Whisper.

## Características

- **Transcripción en tiempo real** de audio del micrófono y Discord
- **Interfaz de chat translúcida** que no interfiere con otras aplicaciones (click-through)
- **Aceleración por GPU** (CUDA) para procesamiento más rápido
- **Optimizado para español** con reconocimiento de alta calidad
- **Eliminación inteligente de "alucinaciones"** y falsos positivos
- **Marcas de tiempo** para cada mensaje
- **Modo compacto** en la esquina inferior derecha
- **Detección de pausas** entre frases

## Requisitos

- Python 3.8+
- PyQt5
- OpenAI Whisper
- PyAudio
- NumPy
- PyTorch (preferiblemente con soporte CUDA)

## Uso

1. Asegúrate de tener todos los requisitos instalados
2. Ejecuta `python device_list.py` para identificar el índice de tu dispositivo de audio
3. Ajusta la variable `MIC_DEVICE` en `discord_whisper_complete.py` si es necesario
4. Ejecuta `python discord_whisper_complete.py`

## Configuración

El archivo principal contiene variables de configuración para personalizar el comportamiento:

- `USE_GPU`: Activar/desactivar aceleración por GPU
- `LANGUAGE`: Idioma para la transcripción ("es" para español)
- `MODEL_SIZE`: Tamaño del modelo Whisper ("tiny", "base", "small", "medium")
- `CHAT_HEIGHT`, `CHAT_WIDTH`: Dimensiones de la ventana
- `POSITION_BOTTOM_RIGHT`: Colocar en la esquina inferior derecha

## Archivos del proyecto

- `discord_whisper_complete.py`: La aplicación principal completa
- `device_list.py`: Utilidad para listar dispositivos de audio disponibles
- `README.md`: Este archivo de documentación

## Licencia

MIT
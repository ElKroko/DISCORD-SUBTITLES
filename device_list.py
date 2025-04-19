#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilidad para listar dispositivos de audio disponibles en el sistema.
Útil para identificar los índices correctos para configurar MIC_DEVICE y DISCORD_DEVICE
en la aplicación Discord Whisper Overlay.
"""

import pyaudio

def list_audio_devices():
    """Listar todos los dispositivos de audio disponibles."""
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    print("\n" + "=" * 80)
    print("DISPOSITIVOS DE AUDIO DISPONIBLES")
    print("=" * 80)
    
    input_devices = []
    output_devices = []
    
    # Recopilar información de dispositivos
    for i in range(num_devices):
        device_info = p.get_device_info_by_index(i)
        device_name = device_info.get('name')
        max_input_channels = device_info.get('maxInputChannels')
        max_output_channels = device_info.get('maxOutputChannels')
        
        device_type = []
        if max_input_channels > 0:
            device_type.append("ENTRADA")
            input_devices.append((i, device_name))
        if max_output_channels > 0:
            device_type.append("SALIDA")
            output_devices.append((i, device_name))
            
        device_type_str = " y ".join(device_type)
        
        print(f"[{i}] {device_name} - Tipo: {device_type_str}")
        print(f"    Canales de entrada: {max_input_channels}")
        print(f"    Canales de salida: {max_output_channels}")
        print(f"    Frecuencia predeterminada: {int(device_info.get('defaultSampleRate'))} Hz")
        print("-" * 80)
    
    # Dispositivo de entrada predeterminado
    try:
        default_input = p.get_default_input_device_info()
        print(f"\nDispositivo de entrada predeterminado: [{default_input['index']}] {default_input['name']}")
    except:
        print("\nNo se pudo detectar el dispositivo de entrada predeterminado")
    
    # Dispositivo de salida predeterminado
    try:
        default_output = p.get_default_output_device_info()
        print(f"Dispositivo de salida predeterminado: [{default_output['index']}] {default_output['name']}")
    except:
        print("No se pudo detectar el dispositivo de salida predeterminado")
    
    # Recomendaciones para configurar la aplicación
    print("\n" + "=" * 80)
    print("RECOMENDACIONES PARA CONFIGURAR DISCORD WHISPER OVERLAY")
    print("=" * 80)
    
    print("\nPara capturar tu voz (micrófono), configura MIC_DEVICE con uno de estos valores:")
    for idx, name in input_devices:
        print(f"  MIC_DEVICE = {idx}  # {name}")
    
    print("\nPara capturar audio de Discord, configura DISCORD_DEVICE con uno de estos valores:")
    for idx, name in output_devices:
        print(f"  DISCORD_DEVICE = {idx}  # {name}")
    
    print("\nNota: Para capturar audio de Discord, idealmente deberías usar VB-Cable u otra")
    print("solución de cable de audio virtual para redirigir el audio de Discord a un dispositivo")
    print("de entrada virtual que pueda ser capturado por esta aplicación.")
    
    # Limpiar
    p.terminate()

if __name__ == "__main__":
    try:
        list_audio_devices()
        print("\nPresiona Enter para salir...")
        input()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPresiona Enter para salir...")
        input()
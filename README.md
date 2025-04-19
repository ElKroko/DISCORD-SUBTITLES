# Discord Whisper Overlay

A real-time speech transcription overlay for Discord conversations that displays both your microphone input and Discord audio as text on screen.

## Features

- Transparent overlay that stays on top of all windows
- Real-time transcription of your microphone input (displayed in red)
- Real-time transcription of Discord audio (displayed in blue)
- Low latency audio processing for near real-time transcription
- Powered by OpenAI's Whisper speech recognition model

## Requirements

- Python 3.8+
- PyAudio (for audio capture)
- OpenAI Whisper (for speech recognition)
- PyQt5 (for the overlay interface)
- NumPy (for audio processing)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/YOUR_USERNAME/discord-whisper-overlay.git
   cd discord-whisper-overlay
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. Install the required packages:
   ```
   pip install numpy pyaudio whisper openai-whisper PyQt5
   ```

## Configuration

You may need to adjust the following variables in `discord_whisper_overlay.py`:

- `MIC_DEVICE`: Your microphone device index or name (default: None - uses system default)
- `DISCORD_DEVICE`: Your Discord audio output device index or name (default: None - uses system default)
- `BUFFER_SECONDS`: Audio buffer size in seconds (default: 2)
- Whisper model size in the `main()` function (options: "tiny", "base", "small", "medium", "large")

## Usage

1. Make sure your virtual environment is activated
2. Run the program:
   ```
   python discord_whisper_overlay.py
   ```
3. A transparent overlay should appear at the top of your screen
4. Your speech will be transcribed in red text, and Discord audio in blue text

## How It Works

The program uses two separate audio capture threads to process:
1. Your microphone input
2. Discord audio output

Each audio stream is processed in real-time through the Whisper model, which converts speech to text. The text is then displayed in the overlay window.

## Troubleshooting

- If no text appears, check your microphone and Discord audio settings
- To identify correct device indices, run `python -c "import pyaudio; p = pyaudio.PyAudio(); [print(p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"`
- Adjust the model size based on your computer's performance capabilities

## License

[MIT License](LICENSE)
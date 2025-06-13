import os
from audio_transcriber import AudioTranscriber

def main():
    # Percorso base due cartelle sopra rispetto a questo file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Percorso file audio
    audio_path = os.path.join(base_dir, "rbg3.m4a")  # Cambia nome se serve

    # Inizializza solo con default (più veloce)
    transcriber = AudioTranscriber()

    # Ottieni solo il testo continuo, senza timestamp
    text = transcriber.transcribe_text_only(audio_path)
    print(text)

if __name__ == "__main__":
    main()

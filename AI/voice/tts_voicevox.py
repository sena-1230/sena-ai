import requests
import sounddevice as sd
from scipy.io import wavfile
import io

VOICEVOX_URL = "http://127.0.0.1:50021"

def speak(text: str, speaker: int = 3):
    q = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=10
    )
    q.raise_for_status()

    wav = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker},
        data=q.content,
        timeout=30
    )
    wav.raise_for_status()

    rate, data = wavfile.read(io.BytesIO(wav.content))
    sd.play(data, rate)
    sd.wait()

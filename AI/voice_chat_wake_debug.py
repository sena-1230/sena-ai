import os, io, wave, json
import requests
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
import time

VOICEVOX_URL = "http://127.0.0.1:50021"
SPEAKER_ID = 3
WAKE_WORDS = ("セナ", "せな", "聖奈", "星奈", "セーナ", "せーな")
STOP_WORDS = ("終了", "ストップ", "やめ", "やめて")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def play_wav_bytes(wav_bytes: bytes) -> None:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        sd.play(audio, wf.getframerate())
        sd.wait()

def speak_voicevox(text: str, speaker: int = SPEAKER_ID) -> None:
    aq = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=30
    )
    aq.raise_for_status()
    query = aq.json()

    # 体感を速くする（好みで1.1〜1.4）
    query["speedScale"] = 1.25
    query["pauseLength"] = 0.1
    query["pauseLengthScale"] = 0.6

    syn = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker},
        data=json.dumps(query),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    syn.raise_for_status()
    play_wav_bytes(syn.content)

def ask_ai(user_text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=60,
        messages=[
            {"role": "system", "content": "日本語で短く会話。返答は1文、最大20文字。"},
            {"role": "user", "content": user_text},
        ],
    )
    return resp.choices[0].message.content.strip()

def extract_command(text: str):
    t = text.strip()

    if any(w in t for w in STOP_WORDS):
        return "__STOP__"

    if not any(w in t for w in WAKE_WORDS):
        return None

    for w in WAKE_WORDS:
        if w in t:
            after = t.split(w, 1)[1]
            after = after.lstrip(" 、,　").strip()
            return after if after else "__WAKE_ONLY__"

    return "__WAKE_ONLY__"

def main():
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("🎛 ノイズ調整中…")
        r.adjust_for_ambient_noise(source, duration=0.8)
        r.pause_threshold = 0.45
        r.non_speaking_duration = 0.25
        r.dynamic_energy_threshold = True

    print("✅ 待機中（起動ワード: 'セナ、〜'）")

    while True:
        try:
            t0 = time.perf_counter()

            print("🎤 聞いてる…")
            with mic as source:
                audio = r.listen(source, timeout=3, phrase_time_limit=4)

            print("🧠 認識中…")
            try:
                text = r.recognize_google(audio, language="ja-JP").strip()
            except sr.UnknownValueError:
                print("…聞き取れなかった（もう一回）")
                continue

            t_stt = time.perf_counter()
            print("👂", text)

            cmd = extract_command(text)
            if cmd is None:
                print("…起動ワードなし → 無視")
                continue

            if cmd == "__STOP__":
                speak_voicevox("了解。終了するね。")
                break

            if cmd == "__WAKE_ONLY__":
                speak_voicevox("はい。どうしたの？")
                continue

            ai_text = ask_ai(cmd)
            t_ai = time.perf_counter()

            print("🤖", ai_text)
            speak_voicevox(ai_text)
            t_tts = time.perf_counter()

            print(f"⏱ STT:{t_stt-t0:.2f}s  AI:{t_ai-t_stt:.2f}s  TTS:{t_tts-t_ai:.2f}s  total:{t_tts-t0:.2f}s")

        except sr.WaitTimeoutError:
            print("…無音（待機継続）")
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("❌ エラー:", e)

    print("👋 終了")

if __name__ == "__main__":
    main()

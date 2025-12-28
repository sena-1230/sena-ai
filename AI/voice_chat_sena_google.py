import os, io, wave, json, time, re, queue, threading
import requests
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI

from memory import (
    load_state,
    save_state,
    maybe_compress_history,
    load_profile,
    build_profile_system_message,
)

# ================= 設定 =================
VOICEVOX_URL = "http://127.0.0.1:50021"
SPEAKER_ID = 1  # お姉さん系（四国めたん）

WAKE_WORD_PATTERN = r"^(セナ|せな|瀬名|聖奈|星奈|せいな|せーな|セーナ)[、, ]*(.*)$"

STOP_WORDS = ("終了", "ストップ", "やめ", "やめて")

MODEL = "gpt-4o-mini"
MAX_TOKENS = 140
SENTENCE_DELIMS = ("。", "！", "？", "!", "?", "\n")
KEEP_LAST_N = 10

# -------- 会話状態 --------
awake = False
last_awake_time = 0
# =======================================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_MESSAGES = [
    {"role": "system", "content": "あなたはセナ。優しく、短めに、わかりやすく日本語で返答します。"}
]

state = load_state(DEFAULT_MESSAGES)
summary = state.get("summary", "")
messages = state.get("messages", DEFAULT_MESSAGES)

profile = load_profile()
profile_system = build_profile_system_message(profile)

# ================= 音声 =================
def play_wav_bytes(wav_bytes: bytes) -> None:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        sd.play(audio, wf.getframerate())
        sd.wait()

def voicevox_synthesize(text: str, speaker: int = SPEAKER_ID) -> bytes:
    aq = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=15,
    )
    aq.raise_for_status()
    query = aq.json()

    # お姉さん・きれいめ調整
    query["speedScale"] = 1.15
    query["pauseLength"] = 0.12
    query["pauseLengthScale"] = 0.7

    syn = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker},
        data=json.dumps(query),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    syn.raise_for_status()
    return syn.content

class TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()

    def run(self):
        while True:
            text = self.q.get()
            if text is None:
                break
            wav = voicevox_synthesize(text)
            play_wav_bytes(wav)

    def say(self, text: str):
        self.q.put(text)

tts = TTSWorker()
tts.start()

# ================= 会話制御 =================
def split_for_tts(buf: str):
    for d in SENTENCE_DELIMS:
        if d in buf:
            i = buf.index(d) + 1
            return buf[:i], buf[i:]
    return None, buf

def extract_command(text: str):
    """
    最初だけ「セナ」で起床。
    起床後は呼びかけ無しでも会話を通す。
    """
    global awake, last_awake_time

    t = text.strip()
    # 呼びかけ単体（表記ゆれ）を強制ウェイク
    if t in ("セナ", "せな", "瀬名", "聖奈", "星奈", "せいな", "せーな", "セーナ"):
        awake = True
        last_awake_time = time.time()
        return "__WAKE_ONLY__"


    # 終了
    if any(w in t for w in STOP_WORDS):
        awake = False
        return "__STOP__"

    # 呼びかけ
    m = re.match(WAKE_WORD_PATTERN, t)
    if m:
        awake = True
        last_awake_time = time.time()
        cmd = (m.group(2) or "").strip()
        return cmd if cmd else "__WAKE_ONLY__"

    # 起床中なら普通に会話
    if awake:
        last_awake_time = time.time()
        return t

    return None

def ask_ai_stream(user_text: str):
    global summary, messages

    messages.append({"role": "user", "content": user_text})

    send_messages = [{"role": "system", "content": profile_system}]
    if summary:
        send_messages.append({"role": "system", "content": summary})
    send_messages.extend(messages[1:])

    stream = client.chat.completions.create(
        model=MODEL,
        messages=send_messages,
        stream=True,
        max_tokens=MAX_TOKENS,
    )

    buf = ""
    print("🤖 セナ: ", end="", flush=True)

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        buf += delta
        part, buf = split_for_tts(buf)
        if part:
            tts.say(part)

    if buf:
        tts.say(buf)

    messages.append({"role": "assistant", "content": buf})
    summary, messages, _ = maybe_compress_history(
        client, summary, messages, KEEP_LAST_N, MODEL
    )
    save_state(summary, messages)

# ================= メイン =================
def main():
    r = sr.Recognizer()
    mic = sr.Microphone()

    print("✅ セナ起動。最初に『セナ』って呼んでね")
    tts.say("起動したよ。セナ、って呼んで。")

    while True:
        with mic as source:
            audio = r.listen(source)

        try:
            text = r.recognize_google(audio, language="ja-JP")
        except:
            continue

        print("👂", text)

        cmd = extract_command(text)

        if not cmd:
            continue

        if cmd == "__STOP__":
            tts.say("終了するね")
            break

        if cmd == "__WAKE_ONLY__":
            tts.say("なに？")
            time.sleep(0.2)
            continue

        ask_ai_stream(cmd)

if __name__ == "__main__":
    main()

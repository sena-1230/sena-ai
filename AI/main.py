import os
from dotenv import load_dotenv
from openai import OpenAI
from voice.tts_voicevox import speak


from memory import (
    load_state,
    save_state,
    reset_state,
    maybe_compress_history,
    load_profile,
    build_profile_system_message,
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 以前のmessages先頭にsystemが入っていてもOKなようにする
DEFAULT_MESSAGES = [
    {"role": "system", "content": "あなたはセナ。優しく、短めに、わかりやすく日本語で返答します。"}
]

# 履歴（summary+messages）読み込み
state = load_state(DEFAULT_MESSAGES)
summary = state.get("summary", "")
messages = state.get("messages", DEFAULT_MESSAGES)

# プロフィール読み込み（人格固定）
profile = load_profile()
profile_system = build_profile_system_message(profile)

KEEP_LAST_N = 10  # まずテストしやすく
MODEL = "gpt-4o-mini"

print("セナ起動しました。終了: exit / リセット: reset / 要約表示: summary")

while True:
    user_text = input("\nあなた: ").strip()

    if user_text.lower() in ["exit", "quit", "q"]:
        save_state(summary, messages)
        print("セナ: またね！")
        break

    if user_text.lower() == "reset":
        summary = ""
        messages = DEFAULT_MESSAGES.copy()
        reset_state(messages)
        print("セナ: 会話をリセットしたよ。")
        continue

    if user_text.lower() == "summary":
        print("\n--- 長期記憶（要約） ---")
        print(summary if summary else "(まだ要約は空だよ)")
        print("----------------------")
        continue

    # ユーザー発言を短期履歴へ
    messages.append({"role": "user", "content": user_text})

    # ✅ 送る内容を組み立て（人格固定 → 長期記憶 → 短期記憶）
    send_messages = []
    send_messages.append({"role": "system", "content": profile_system})
    if summary:
        send_messages.append({"role": "system", "content": f"【長期記憶（要約）】\n{summary}"})

    # messages の先頭が system なら二重になるので除外
    if messages and messages[0].get("role") == "system":
        send_messages.extend(messages[1:])
    else:
        send_messages.extend(messages)

    # API呼び出し
    # API呼び出し
    # API呼び出し
    response = client.chat.completions.create(
        model=MODEL,
        messages=send_messages,
    )

    # 返答テキストを取り出す
    assistant_text = response.choices[0].message.content or ""

    # 表示
    print(f"セナ: {assistant_text}")

    # 音声で読み上げ（文字列を渡す）
    print("DEBUG: speak呼び出し直前")
    try:
        if assistant_text.strip():
            speak(assistant_text)
        else:
            print("DEBUG: assistant_textが空でした")
        print("DEBUG: speak呼び出し完了")
    except Exception as e:
        print("DEBUG: speakでエラー:", e)

    # 返答を短期履歴へ
    messages.append({"role": "assistant", "content": assistant_text})

    # 長くなったら要約して圧縮
    summary, messages, compressed = maybe_compress_history(
        client=client,
        summary=summary,
        messages=messages,
        keep_last_n=KEEP_LAST_N,
        model=MODEL,
    )

    # 保存
    save_state(summary, messages)

    if compressed:
        print("(セナ: ちょっと昔の会話を要約して覚え直したよ)")

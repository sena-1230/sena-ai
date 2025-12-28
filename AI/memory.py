import json
from pathlib import Path
from typing import Any

HISTORY_PATH = Path("history.json")
PROFILE_PATH = Path("profile.json")
def load_profile() -> dict:
    """
    profile.json を読み込む（無ければ空プロフィール）
    """
    if not PROFILE_PATH.exists():
        return {}

    try:
        text = PROFILE_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_profile_system_message(profile: dict) -> str:
    """
    profile.json を system 文（人格固定）に変換する
    """
    name = profile.get("assistant_name", "セナ")
    tone = profile.get("tone", "優しく短めに日本語で")
    role = profile.get("role", "ユーザーを支えるAI")

    principles = profile.get("principles", [])
    do_not = profile.get("do_not", [])
    user_context = profile.get("user_context", {})

    p_text = "\n".join([f"- {p}" for p in principles]) if principles else "- なし"
    n_text = "\n".join([f"- {n}" for n in do_not]) if do_not else "- なし"
    u_text = json.dumps(user_context, ensure_ascii=False, indent=2) if user_context else "{}"

    return (
        f"あなたの名前は{name}。\n"
        f"役割: {role}\n"
        f"口調/スタイル: {tone}\n\n"
        f"行動原則:\n{p_text}\n\n"
        f"禁止事項:\n{n_text}\n\n"
        f"ユーザー前提（参考）:\n{u_text}\n"
    )


def load_state(default_messages: list[dict]) -> dict:
    """
    history.json から state を読み込む。
    state 形式:
      {
        "summary": "これまでの要約（長期記憶）",
        "messages": [ ...直近の会話... ]
      }
    """
    default_state = {"summary": "", "messages": default_messages}

    if not HISTORY_PATH.exists():
        return default_state

    try:
        text = HISTORY_PATH.read_text(encoding="utf-8")
        data = json.loads(text)

        if not isinstance(data, dict):
            return default_state

        summary = data.get("summary", "")
        messages = data.get("messages", default_messages)

        if not isinstance(summary, str):
            summary = ""
        if not isinstance(messages, list):
            messages = default_messages

        # 最低限の形式チェック
        for item in messages:
            if not isinstance(item, dict):
                return default_state
            if "role" not in item or "content" not in item:
                return default_state

        return {"summary": summary, "messages": messages}

    except Exception:
        return default_state


def save_state(summary: str, messages: list[dict]) -> None:
    """
    state を history.json に保存する。
    """
    payload = {"summary": summary, "messages": messages}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    HISTORY_PATH.write_text(text, encoding="utf-8")


def reset_state(default_messages: list[dict]) -> None:
    save_state("", default_messages)


def build_summary_prompt(old_summary: str, older_messages: list[dict]) -> list[dict]:
    """
    要約を作るための messages（プロンプト）を作る。
    old_summary と older_messages を材料にして、新しい summary を生成させる。
    """
    # 文字数を増やしすぎないため、必要最低限で圧縮する指示にする
    instruction = (
        "あなたは会話ログを要約する担当です。\n"
        "目的: 長期記憶として使える要約を作ること。\n"
        "制約:\n"
        "- 日本語で。\n"
        "- 200〜500文字程度。\n"
        "- 固有名詞・重要な好み・目標・決定事項・未解決タスクを優先。\n"
        "- 余計な推測はしない。\n"
        "- 箇条書き多めで読みやすく。\n"
    )

    # older_messages をそのまま渡すと長くなるので、role/contentだけに限定
    compact = [{"role": m["role"], "content": m["content"]} for m in older_messages]

    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"これまでの要約（既存）:\n{old_summary}\n\n新しく追加して要約したい過去会話:\n{json.dumps(compact, ensure_ascii=False)}\n\n更新後の要約を作ってください。"}
    ]


def maybe_compress_history(
    client: Any,
    summary: str,
    messages: list[dict],
    keep_last_n: int = 30,
    model: str = "gpt-4o-mini",
) -> tuple[str, list[dict], bool]:
    """
    messages が長くなりすぎたら、古い部分を要約して summary に吸収し、
    messages は直近 keep_last_n 件だけ残す。

    戻り値: (new_summary, new_messages, compressed?)
    """
    if len(messages) <= keep_last_n:
        return summary, messages, False

    older = messages[:-keep_last_n]
    recent = messages[-keep_last_n:]

    prompt = build_summary_prompt(summary, older)

    resp = client.chat.completions.create(
        model=model,
        messages=prompt,
    )

    new_summary = resp.choices[0].message.content.strip()
    return new_summary, recent, True

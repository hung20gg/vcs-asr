import json
from collections import defaultdict

SESSION_SPLIT = 60.0          # tách cuộc hội thoại mới
INTRA_SPEAKER_SPLIT = 5.0    # tách đoạn của cùng 1 người


def load_logs(file_path):
    logs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def split_sessions(logs, threshold):
    sessions = []
    current = []
    last_end = None

    for item in logs:
        if last_end is not None:
            if item["start"] - last_end > threshold:
                sessions.append(current)
                current = []

        current.append(item)
        last_end = item["end"]

    if current:
        sessions.append(current)

    return sessions


def group_by_speaker_with_pause(session):
    """
    Trả về list các block:
    [
      {speaker: "...", texts: [...]},
      ...
    ]
    """
    blocks = []

    prev_speaker = None
    prev_end = None

    for item in session:
        speaker = item.get("speaker", "unknown")
        text = item.get("text", "").strip()
        start = item["start"]

        if not text:
            continue

        # tạo block mới nếu:
        # - khác speaker
        # - hoặc cùng speaker nhưng pause quá lâu
        if (
            speaker != prev_speaker or
            (prev_end is not None and start - prev_end > INTRA_SPEAKER_SPLIT)
        ):
            blocks.append({
                "speaker": speaker,
                "texts": [text]
            })
        else:
            blocks[-1]["texts"].append(text)

        prev_speaker = speaker
        prev_end = item["end"]

    return blocks


def to_markdown(sessions):
    md = []

    for i, session in enumerate(sessions, 1):
        md.append(f"## Cuộc hội thoại {i}\n")

        blocks = group_by_speaker_with_pause(session)

        for block in blocks:
            speaker = block["speaker"]
            combined = " ".join(block["texts"])
            md.append(f"### 👤 {speaker}")
            md.append(f"- {combined}\n")

    return "\n".join(md)


if __name__ == "__main__":
    file_path = "test/log/transcribe_stream_2.jsonl"

    logs = load_logs(file_path)
    sessions = split_sessions(logs, SESSION_SPLIT)
    markdown_output = to_markdown(sessions)

    with open("transcript.md", "w", encoding="utf-8") as f:
        f.write(markdown_output)
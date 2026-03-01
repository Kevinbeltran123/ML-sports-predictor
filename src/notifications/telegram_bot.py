"""
TelegramBot: wrapper de envio de mensajes via Bot API.

Usa requests directamente (ya en requirements.txt).
Config via env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
import os
import time
import requests
from src.config import get_logger

logger = get_logger(__name__)

MAX_MSG_LEN = 4096


def _chunk_message(text: str, max_len: int) -> list[str]:
    """Split text into chunks on paragraph boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If single paragraph exceeds max, split on newlines
            if len(paragraph) > max_len:
                lines = paragraph.split("\n")
                current = ""
                for line in lines:
                    candidate = f"{current}\n{line}" if current else line
                    if len(candidate) <= max_len:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = line[:max_len]
            else:
                current = paragraph
    if current:
        chunks.append(current)
    return chunks


class TelegramBot:
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = (token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
        self._base = f"https://api.telegram.org/bot{self.token}"
        if not self.token or not self.chat_id:
            logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — dry-run mode")

    def _send_raw(self, text: str, parse_mode: str = "MarkdownV2") -> bool:
        """POST to sendMessage. Returns True on success."""
        if not self.token or not self.chat_id:
            logger.info("[TELEGRAM DRY-RUN] %s", text[:200])
            return True

        payload = {
            "chat_id": self.chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self._base}/sendMessage",
                    json=payload,
                    timeout=10,
                )
                if resp.status_code == 200:
                    return True
                # MarkdownV2 parse errors — fallback to plain text
                if resp.status_code == 400 and "parse" in resp.text.lower() and parse_mode:
                    logger.warning("MarkdownV2 parse error, retrying as plain text")
                    payload.pop("parse_mode", None)
                    resp2 = requests.post(
                        f"{self._base}/sendMessage",
                        json=payload,
                        timeout=10,
                    )
                    if resp2.status_code == 200:
                        return True
                    logger.warning("Plain text fallback also failed: %s", resp2.text[:200])
                else:
                    logger.warning("Telegram API %d: %s", resp.status_code, resp.text[:200])
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("Telegram send error (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        return False

    def send(self, text: str, parse_mode: str = "MarkdownV2") -> None:
        """Send text, chunking if > 4096 chars."""
        if len(text) <= MAX_MSG_LEN:
            self._send_raw(text, parse_mode)
            return
        chunks = _chunk_message(text, MAX_MSG_LEN)
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk = "_(cont\\.)_\n" + chunk if parse_mode == "MarkdownV2" else "(cont.)\n" + chunk
            self._send_raw(chunk, parse_mode)
            time.sleep(0.5)  # avoid flood limits

    def send_plain(self, text: str) -> None:
        """Send plain text (no markdown parsing)."""
        if len(text) <= MAX_MSG_LEN:
            self._send_raw(text, parse_mode="")
            return
        chunks = _chunk_message(text, MAX_MSG_LEN)
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk = "(cont.)\n" + chunk
            self._send_raw(chunk, parse_mode="")
            time.sleep(0.5)

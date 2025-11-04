from __future__ import annotations

import os
import time
from typing import Any, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from app.observability.logging import get_logger


logger = get_logger("slack")


class SlackAdapter:
    def __init__(self, token: str | None = None, default_channel: str | None = None):
        self.client = WebClient(token=token or os.getenv("SLACK_BOT_TOKEN"))
        self.default_channel = default_channel or os.getenv("SLACK_CHANNEL", "all-trading")
        self.enabled = bool(self.client.token)
        self._threads: dict[str, str] = {}

    def post(self, text: str, channel: str | None = None, blocks: list[dict[str, Any]] | None = None) -> None:
        if not self.enabled:
            logger.info("slack.demo.post", text=text)
            return
        try:
            self.client.chat_postMessage(channel=channel or self.default_channel, text=text, blocks=blocks)
        except SlackApiError as e:
            logger.error("slack.error", err=str(e))

    def _post_thread_message(
        self,
        thread_key: str,
        text: str,
        blocks: list[dict[str, Any]],
        mention: Optional[str] = None,
        ping: bool = False,
    ) -> str:
        if mention and ping:
            text = f"{mention} {text}"

        if not self.enabled:
            logger.info("slack.demo.thread", key=thread_key, text=text, blocks=blocks)
            synthetic_ts = f"demo-{thread_key}-{int(time.time())}"
            if thread_key not in self._threads:
                self._threads[thread_key] = synthetic_ts
            return synthetic_ts

        try:
            if thread_key not in self._threads:
                resp = self.client.chat_postMessage(
                    channel=self.default_channel,
                    text=text,
                    blocks=blocks,
                )
                self._threads[thread_key] = resp["ts"]
                return resp["ts"]
            else:
                thread_ts = self._threads[thread_key]
                resp = self.client.chat_postMessage(
                    channel=self.default_channel,
                    text=text,
                    blocks=blocks,
                    thread_ts=thread_ts,
                )
                return resp["ts"]
        except SlackApiError as e:
            logger.error("slack.error", err=str(e), thread=thread_key)
        return self._threads.get(thread_key, thread_key)

    def post_heads_up(self, symbol: str, box, metrics: dict[str, Any], thread_key: str, mention: Optional[str] = None) -> None:
        text = f"HEADS-UP · {symbol} · {box.bars}-bar box {box.lo:.2f}–{box.hi:.2f}"
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{symbol}* · {box.bars}-bar {box.tf}\n"
                        f"Box: {box.lo:.2f} – {box.hi:.2f} (h={box.height:.3f})\n"
                        f"Spread: {box.spread_cents:.1f}¢ · RVOL: {metrics.get('rvol', box.rvol):.2f}\n"
                        f"Nearest level: {metrics.get('nearest_level', 'n/a')}"
                    ),
                },
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Quality: {box.quality_score:.2f}"},
                    {"type": "mrkdwn", "text": f"TTL: {metrics.get('ttl', 'n/a')}"},
                    {"type": "mrkdwn", "text": metrics.get("notes", "")},
                ],
            },
        ]
        return self._post_thread_message(thread_key, text, blocks, mention=mention, ping=False)

    def post_primed(
        self,
        thread_key: str,
        symbol: str,
        direction: str,
        payload: dict[str, Any],
        mention: Optional[str],
        ping: bool,
    ) -> None:
        header = {
            "type": "header",
            "text": {"type": "plain_text", "text": f"PRIMED {direction.upper()} · {symbol}", "emoji": True},
        }
        body = {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Entry* {payload.get('entry', 'n/a')}"},
                {"type": "mrkdwn", "text": f"*Stop* {payload.get('stop', 'n/a')}"},
                {"type": "mrkdwn", "text": f"*Target* {payload.get('target', 'n/a')}"},
                {"type": "mrkdwn", "text": f"*R:R* {payload.get('rr', 'n/a')}"},
                {"type": "mrkdwn", "text": f"*p2R* {payload.get('p2r', 'n/a')}"},
                {"type": "mrkdwn", "text": f"*L2%* {payload.get('l2', 'n/a')}"},
                {"type": "mrkdwn", "text": f"*Spread* {payload.get('spread', 'n/a')}"},
            ],
        }
        context = {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"TTL: {payload.get('ttl', 'n/a')}"},
                {"type": "mrkdwn", "text": payload.get("note", "")},
            ],
        }
        blocks = [header, body, context]
        return self._post_thread_message(thread_key, f"PRIMED {direction.upper()} · {symbol}", blocks, mention=mention, ping=ping)

    def post_trigger(
        self,
        thread_key: str,
        symbol: str,
        payload: dict[str, Any],
        mention: Optional[str],
        ping: bool,
    ) -> None:
        header = {
            "type": "header",
            "text": {"type": "plain_text", "text": f"BREAK + RETEST LIVE · {symbol}", "emoji": True},
        }
        body = {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Entry* {payload.get('entry')}"},
                {"type": "mrkdwn", "text": f"*Stop* {payload.get('stop')}"},
                {"type": "mrkdwn", "text": f"*Targets* {payload.get('targets')}"},
                {"type": "mrkdwn", "text": f"*R:R* {payload.get('rr')}"},
                {"type": "mrkdwn", "text": f"*p2R* {payload.get('p2r', 'n/a')}"},
                {"type": "mrkdwn", "text": f"*Spread* {payload.get('spread')}"},
                {"type": "mrkdwn", "text": f"*RVOL* {payload.get('rvol')}"},
            ],
        }
        context = {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"L2%: {payload.get('l2', 'n/a')}"},
                {"type": "mrkdwn", "text": f"Box: {payload.get('box_summary')}"},
                *([{ "type": "mrkdwn", "text": payload.get("note", "") }] if payload.get("note") else []),
            ],
        }
        actions = {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Open TradingView"},
                    "url": payload.get("chart_url", "https://www.tradingview.com"),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Acknowledge"},
                    "value": "acknowledge",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Pass"},
                    "value": "pass",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Discuss"},
                    "value": "discuss",
                },
            ],
        }
        blocks = [header, body, context, actions]
        return self._post_thread_message(thread_key, f"BREAK + RETEST LIVE · {symbol}", blocks, mention=mention, ping=ping)

    def update_message(self, message_ts: str, blocks: list[dict[str, Any]], text: str) -> None:
        if not self.enabled:
            logger.info("slack.demo.update", ts=message_ts, text=text)
            return
        try:
            self.client.chat_update(channel=self.default_channel, ts=message_ts, text=text, blocks=blocks)
        except SlackApiError as e:
            logger.error("slack.error", err=str(e), ts=message_ts)

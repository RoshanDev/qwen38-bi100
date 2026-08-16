#!/usr/bin/env python3
"""Expose an OpenAI Responses-compatible endpoint over a Chat Completions model.

The bridge is intentionally dependency-free. It is aimed at Codex CLI clients
whose custom providers require the Responses API, while the upstream vLLM
server only implements Chat Completions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
TOOL_PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_value(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"input_text", "output_text", "text"}:
            parts.append(str(item.get("text", "")))
        elif item_type in {"input_image", "image_url"}:
            parts.append("[image input omitted: this deployment is text-only]")
    return "\n".join(part for part in parts if part)


def responses_input_to_chat(body: dict[str, Any]) -> list[dict[str, Any]]:
    system_parts: list[str] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        system_parts.append(instructions.strip())

    messages: list[dict[str, Any]] = []
    input_value = body.get("input", [])
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        input_items: list[Any] = []
    elif isinstance(input_value, list):
        input_items = input_value
    else:
        input_items = []

    for item in input_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "message")

        if item_type == "message":
            role = str(item.get("role", "user"))
            text = _content_text(item.get("content"))
            if role in {"system", "developer"}:
                if text:
                    system_parts.append(text)
            elif role in {"user", "assistant", "tool"}:
                messages.append({"role": role, "content": text})
            continue

        if item_type == "function_call":
            name = str(item.get("name", ""))
            arguments = item.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": str(item.get("call_id") or _new_id("call")),
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            )
            continue

        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id", "")),
                    "content": _content_text(item.get("output")),
                }
            )

    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    if not messages:
        messages.append({"role": "user", "content": ""})
    return messages


def responses_tools_to_chat(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        name = tool.get("name")
        if tool_type == "function" and isinstance(name, str):
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(tool.get("description", "")),
                        "parameters": tool.get("parameters")
                        if isinstance(tool.get("parameters"), dict)
                        else {"type": "object", "properties": {}},
                    },
                }
            )
        elif tool_type == "custom" and isinstance(name, str):
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(tool.get("description", "")),
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string"}},
                            "required": ["input"],
                        },
                    },
                }
            )
    return converted


def tools_as_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Describe tools without using vLLM's tool-choice request fields.

    CoreX's bundled vLLM predates Qwen3.5 tool parsing and rejects both
    automatic and disabled tool_choice values. Qwen still understands its
    native XML tool-call format when the schemas are present in the prompt.
    """
    if not tools:
        return ""

    schemas = "\n".join(
        json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tools
    )
    return (
        "# Tools\n"
        "You have access to the following functions:\n"
        "<tools>\n"
        f"{schemas}\n"
        "</tools>\n\n"
        "If you call a function, reply with only this XML form:\n"
        "<tool_call>\n"
        "<function=FUNCTION_NAME>\n"
        "<parameter=ARGUMENT_NAME>\n"
        "ARGUMENT_VALUE\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "Use one <parameter=...> block for each argument. Do not invent tools."
    )


def responses_request_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    messages = responses_input_to_chat(body)
    tools = responses_tools_to_chat(body.get("tools"))
    tool_prompt = tools_as_system_prompt(tools)
    if tool_prompt:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{tool_prompt}"
        else:
            messages.insert(0, {"role": "system", "content": tool_prompt})

    payload: dict[str, Any] = {
        "model": str(body.get("model", "Qwen3.8-27B")),
        "messages": messages,
        "stream": False,
        "temperature": float(body.get("temperature", 0.0)),
        "max_tokens": int(body.get("max_output_tokens") or 1024),
        "chat_template_kwargs": {"enable_thinking": False},
    }

    if isinstance(body.get("top_p"), (int, float)):
        payload["top_p"] = body["top_p"]

    return payload


def parse_qwen_tool_calls(text: str) -> tuple[str, list[dict[str, str]]]:
    calls: list[dict[str, str]] = []
    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1).strip()
        parameters: dict[str, Any] = {}
        for parameter in TOOL_PARAMETER_RE.finditer(match.group(2)):
            parameters[parameter.group(1).strip()] = _json_value(parameter.group(2))
        calls.append(
            {
                "call_id": _new_id("call"),
                "name": name,
                "arguments": json.dumps(parameters, ensure_ascii=False),
            }
        )

    remaining = TOOL_CALL_RE.sub("", text)
    remaining = THINK_RE.sub("", remaining).strip()
    return remaining, calls


def chat_response_to_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("upstream response has no choices")
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        raise ValueError("upstream choice has no message")

    text = str(message.get("content") or "")
    clean_text, tagged_calls = parse_qwen_tool_calls(text)
    items: list[dict[str, Any]] = []

    if clean_text:
        items.append(
            {
                "id": _new_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": clean_text,
                        "annotations": [],
                    }
                ],
            }
        )

    structured_calls = message.get("tool_calls")
    if isinstance(structured_calls, list):
        for call in structured_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tagged_calls.append(
                {
                    "call_id": str(call.get("id") or _new_id("call")),
                    "name": str(function.get("name", "")),
                    "arguments": arguments,
                }
            )

    for call in tagged_calls:
        items.append(
            {
                "id": _new_id("fc"),
                "type": "function_call",
                "status": "completed",
                "call_id": call["call_id"],
                "name": call["name"],
                "arguments": call["arguments"],
            }
        )

    if not items:
        items.append(
            {
                "id": _new_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        )
    return items


def make_response(body: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    usage = upstream.get("usage", {})
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "id": _new_id("resp"),
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "model": str(body.get("model", "Qwen3.8-27B")),
        "output": chat_response_to_items(upstream),
        "parallel_tool_calls": bool(body.get("parallel_tool_calls", True)),
        "previous_response_id": body.get("previous_response_id"),
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": body.get("temperature", 0.0),
        "text": {"format": {"type": "text"}},
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools", []),
        "top_p": body.get("top_p", 1.0),
        "truncation": body.get("truncation", "disabled"),
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
        "user": None,
        "metadata": body.get("metadata", {}),
    }


def response_events(response: dict[str, Any]) -> list[dict[str, Any]]:
    sequence = 0
    events: list[dict[str, Any]] = []

    def add(event_type: str, **values: Any) -> None:
        nonlocal sequence
        events.append({"type": event_type, "sequence_number": sequence, **values})
        sequence += 1

    in_progress = {**response, "status": "in_progress", "output": []}
    add("response.created", response=in_progress)
    add("response.in_progress", response=in_progress)

    for output_index, item in enumerate(response["output"]):
        add("response.output_item.added", output_index=output_index, item={**item, "status": "in_progress"})
        if item["type"] == "message":
            part = item["content"][0]
            add(
                "response.content_part.added",
                item_id=item["id"],
                output_index=output_index,
                content_index=0,
                part={"type": "output_text", "text": "", "annotations": []},
            )
            add(
                "response.output_text.delta",
                item_id=item["id"],
                output_index=output_index,
                content_index=0,
                delta=part["text"],
            )
            add(
                "response.output_text.done",
                item_id=item["id"],
                output_index=output_index,
                content_index=0,
                text=part["text"],
            )
            add(
                "response.content_part.done",
                item_id=item["id"],
                output_index=output_index,
                content_index=0,
                part=part,
            )
        elif item["type"] == "function_call":
            add(
                "response.function_call_arguments.delta",
                item_id=item["id"],
                output_index=output_index,
                delta=item["arguments"],
            )
            add(
                "response.function_call_arguments.done",
                item_id=item["id"],
                output_index=output_index,
                arguments=item["arguments"],
            )
        add("response.output_item.done", output_index=output_index, item=item)

    add("response.completed", response=response)
    return events


@dataclass(frozen=True)
class Settings:
    upstream_base_url: str
    upstream_api_key: str
    timeout_seconds: int
    use_system_proxy: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            upstream_base_url=os.environ.get(
                "QWEN_UPSTREAM_BASE_URL", "http://127.0.0.1:1111/v1"
            ).rstrip("/"),
            upstream_api_key=os.environ.get("QWEN_UPSTREAM_API_KEY", ""),
            timeout_seconds=int(os.environ.get("QWEN_UPSTREAM_TIMEOUT_SECONDS", "3600")),
            use_system_proxy=os.environ.get("QWEN_UPSTREAM_USE_SYSTEM_PROXY", "0").lower()
            in {"1", "true", "yes", "on"},
        )

    @property
    def chat_url(self) -> str:
        return f"{self.upstream_base_url}/chat/completions"


def call_upstream(settings: Settings, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    request = urllib_request.Request(
        settings.chat_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        opener = (
            urllib_request.build_opener()
            if settings.use_system_proxy
            else urllib_request.build_opener(urllib_request.ProxyHandler({}))
        )
        with opener.open(request, timeout=settings.timeout_seconds) as response:
            return json.load(response)
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"upstream HTTP {exc.code}: {error_body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"failed to reach upstream: {exc.reason}") from exc


class BridgeHandler(BaseHTTPRequestHandler):
    server: "BridgeServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format_string % args}", flush=True)

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = (json.dumps(body, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        value = json.loads(data.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"/healthz", "/v1/healthz"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "upstream": self.server.settings.upstream_base_url,
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in {"/responses", "/v1/responses"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return

        started = time.perf_counter()
        try:
            body = self._read_json()
            payload = responses_request_to_chat(body)
            upstream = call_upstream(self.server.settings, payload)
            response = make_response(body, upstream)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            return
        except RuntimeError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": {"message": str(exc)}})
            return
        except Exception as exc:  # defensive boundary for a local compatibility service
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": str(exc)}})
            return

        elapsed = time.perf_counter() - started
        print(
            "request_ok "
            f"model={body.get('model')} "
            f"input_items={len(body.get('input', [])) if isinstance(body.get('input'), list) else 1} "
            f"output_items={len(response['output'])} elapsed={elapsed:.3f}s",
            flush=True,
        )

        if body.get("stream") is True:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in response_events(response):
                event_type = event["type"]
                data = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"event: {event_type}\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.close_connection = True
            return

        self._send_json(HTTPStatus.OK, response)


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], settings: Settings):
        self.settings = settings
        super().__init__(address, BridgeHandler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("QWEN_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("QWEN_BRIDGE_PORT", "8348"))
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    server = BridgeServer((args.host, args.port), settings)
    print(
        f"qwen responses bridge listening on http://{args.host}:{args.port}/v1 "
        f"upstream={settings.upstream_base_url}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

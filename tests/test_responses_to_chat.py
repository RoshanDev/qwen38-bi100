import json
import unittest

from bridge.responses_to_chat import (
    chat_response_to_items,
    clamp_output_tokens,
    compact_codex_instructions,
    prior_function_call_count,
    parse_qwen_tool_calls,
    responses_request_to_chat,
    trim_tool_messages,
)


class ResponsesBridgeTests(unittest.TestCase):
    def test_tools_are_prompted_but_not_sent_to_legacy_vllm(self) -> None:
        payload = responses_request_to_chat(
            {
                "model": "Qwen3.8-27B",
                "instructions": "Be concise.",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "Run a command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    }
                ],
            }
        )

        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertIn("# Tools", payload["messages"][0]["content"])
        self.assertIn('"name":"exec_command"', payload["messages"][0]["content"])

    def test_qwen_xml_tool_call_is_parsed(self) -> None:
        text, calls = parse_qwen_tool_calls(
            "<tool_call>\n"
            "<function=exec_command>\n"
            "<parameter=cmd>\n\"printf OK\"\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        self.assertEqual("", text)
        self.assertEqual("exec_command", calls[0]["name"])
        self.assertEqual({"cmd": "printf OK"}, json.loads(calls[0]["arguments"]))

    def test_chat_text_becomes_responses_message(self) -> None:
        items = chat_response_to_items(
            {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        )

        self.assertEqual("message", items[0]["type"])
        self.assertEqual("OK", items[0]["content"][0]["text"])

    def test_output_budget_fits_eight_k_context(self) -> None:
        self.assertEqual(731, clamp_output_tokens(1024, 7397, 8192, 64))

    def test_output_budget_rejects_full_context(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "prompt is too large"):
            clamp_output_tokens(1024, 8150, 8192, 64)

    def test_long_codex_instructions_are_compacted(self) -> None:
        original = "You are a coding agent running in the Codex CLI. " + ("x" * 3000)
        compacted = compact_codex_instructions(original, True)
        self.assertLess(len(compacted), 800)
        self.assertIn("coding agent", compacted)
        self.assertIn("at most one relevant file", compacted)

    def test_non_codex_instructions_are_preserved(self) -> None:
        original = "Custom client instructions " + ("x" * 3000)
        self.assertEqual(original, compact_codex_instructions(original, True))

    def test_long_tool_output_is_trimmed(self) -> None:
        messages = [
            {"role": "user", "content": "inspect"},
            {"role": "tool", "content": "A" * 3000 + "THE_END"},
        ]
        self.assertTrue(trim_tool_messages(messages, 2000))
        self.assertIn("tool output truncated", messages[1]["content"])
        self.assertTrue(messages[1]["content"].endswith("THE_END"))
        self.assertLess(len(messages[1]["content"]), 2100)

    def test_prior_tool_calls_are_counted(self) -> None:
        body = {
            "input": [
                {"type": "message", "role": "user", "content": "inspect"},
                {"type": "function_call", "name": "exec_command"},
                {"type": "function_call_output", "output": "ok"},
                {"type": "function_call", "name": "exec_command"},
            ]
        }
        self.assertEqual(2, prior_function_call_count(body))

    def test_tools_can_be_disabled_after_round_limit(self) -> None:
        body = {
            "instructions": "answer",
            "input": "hello",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {"type": "object"},
                }
            ],
        }
        payload = responses_request_to_chat(body, allow_tools=False)
        self.assertNotIn("# Tools", payload["messages"][0]["content"])
        self.assertEqual(4096, payload["max_tokens"])


if __name__ == "__main__":
    unittest.main()

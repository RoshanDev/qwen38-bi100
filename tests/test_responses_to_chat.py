import json
import unittest

from bridge.responses_to_chat import (
    chat_response_to_items,
    parse_qwen_tool_calls,
    responses_request_to_chat,
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


if __name__ == "__main__":
    unittest.main()

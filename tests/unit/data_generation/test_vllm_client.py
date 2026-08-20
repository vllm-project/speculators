import json

from speculators.data_generation.vllm_client import (
    _chat_extra_body,
    stringify_tool_call_arguments,
)

TOOLS = [
    {
        "type": "function",
        "function": {"name": "click", "parameters": {"type": "object"}},
    }
]


class TestChatExtraBody:
    def test_rerender_flags(self):
        body = _chat_extra_body("some/model", None)

        assert body["add_generation_prompt"] is False
        assert body["continue_final_message"] is False
        assert body["return_token_ids"] is True

    def test_mistral_leaves_final_turn_open(self):
        body = _chat_extra_body("mistralai/Mistral-Small", None)

        assert body["continue_final_message"] is True

    def test_tools_sent_with_tool_choice_none(self):
        body = _chat_extra_body("some/model", TOOLS)

        assert body["tools"] == TOOLS
        assert body["tool_choice"] == "none"

    def test_tools_omitted_when_absent(self):
        body = _chat_extra_body("some/model", None)

        assert "tools" not in body
        assert "tool_choice" not in body


class TestStringifyToolCallArguments:
    def test_dict_arguments_become_json_strings(self):
        arguments = {"x": 772, "y": 512}
        messages = [
            {
                "role": "assistant",
                "content": "Clicking.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "click", "arguments": arguments},
                    }
                ],
            }
        ]

        out = stringify_tool_call_arguments(messages)

        sent = out[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(sent, str)
        assert json.loads(sent) == arguments
        # The input is not mutated
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == arguments

    def test_string_arguments_and_plain_turns_pass_through(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "Clicking.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "click", "arguments": '{"x": 65}'},
                    }
                ],
            },
        ]

        assert stringify_tool_call_arguments(messages) == messages

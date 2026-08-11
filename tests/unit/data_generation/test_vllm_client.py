from speculators.data_generation.vllm_client import _chat_extra_body

TOOLS = [
    {
        "type": "function",
        "function": {"name": "click", "parameters": {"type": "object"}},
    }
]


class TestChatExtraBody:
    def test_rerender_flags(self):
        body = _chat_extra_body("some/model", {"input_ids": [1], "messages": []})

        assert body["add_generation_prompt"] is False
        assert body["continue_final_message"] is False
        assert body["return_token_ids"] is True

    def test_mistral_leaves_final_turn_open(self):
        body = _chat_extra_body(
            "mistralai/Mistral-Small", {"input_ids": [1], "messages": []}
        )

        assert body["continue_final_message"] is True

    def test_tools_are_sent(self):
        """The chat template renders tools into the prompt, so a conversation
        tokenized with tools only reproduces its input_ids when they're sent."""
        body = _chat_extra_body(
            "some/model", {"input_ids": [1], "messages": [], "tools": TOOLS}
        )

        assert body["tools"] == TOOLS
        # With tools present, tool_choice defaults to "auto", which vLLM
        # rejects unless the server runs a tool-call parser.
        assert body["tool_choice"] == "none"

    def test_tools_omitted_when_absent(self):
        body = _chat_extra_body("some/model", {"input_ids": [1], "messages": []})

        assert "tools" not in body
        assert "tool_choice" not in body

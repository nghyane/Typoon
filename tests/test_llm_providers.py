"""Tests for provider serialization (OpenAI, Anthropic, Gemini)."""

from typoon.llm.ir import ContentPart, Message, ToolCallMsg, ToolDef


class TestOpenAI:
    def test_system_message(self):
        from typoon.llm.openai import _serialize_message
        out = _serialize_message(Message.system("you are a translator"))
        assert out == {"role": "system", "content": "you are a translator"}

    def test_user_text(self):
        from typoon.llm.openai import _serialize_message
        out = _serialize_message(Message.user_text("translate this"))
        assert out == {"role": "user", "content": "translate this"}

    def test_user_image(self):
        from typoon.llm.openai import _serialize_message
        m = Message.user_parts([
            ContentPart.of_text("look"),
            ContentPart.of_image("data:image/jpeg;base64,abc"),
        ])
        out = _serialize_message(m)
        assert out["role"] == "user"
        assert len(out["content"]) == 2
        assert out["content"][1]["type"] == "image_url"
        assert out["content"][1]["image_url"]["detail"] == "low"

    def test_assistant_tool_calls(self):
        from typoon.llm.openai import _serialize_message
        tc = ToolCallMsg(id="c1", name="translate", arguments='{"x":1}')
        out = _serialize_message(Message.assistant(text=None, tool_calls=[tc]))
        assert out["role"] == "assistant"
        assert len(out["tool_calls"]) == 1
        assert out["tool_calls"][0]["function"]["name"] == "translate"

    def test_tool_result(self):
        from typoon.llm.openai import _serialize_message
        out = _serialize_message(Message.tool_result_text("c1", "ok (3 bubbles)"))
        assert out == {"role": "tool", "tool_call_id": "c1", "content": "ok (3 bubbles)"}

    def test_tool_serialize(self):
        from typoon.llm.openai import _serialize_tool
        t = ToolDef(name="translate", description="Submit", parameters={"type": "object", "properties": {}})
        out = _serialize_tool(t)
        assert out["type"] == "function"
        assert out["function"]["name"] == "translate"
        assert "strict" not in out["function"]


class TestAnthropic:
    def test_system_extraction(self):
        from typoon.llm.anthropic import _extract_system
        msgs = [Message.system("you are X"), Message.user_text("go")]
        out = _extract_system(msgs)
        assert out is not None
        assert out[0]["type"] == "text"
        assert out[0]["cache_control"] == {"type": "ephemeral"}

    def test_system_skipped_in_messages(self):
        from typoon.llm.anthropic import _serialize_messages
        msgs = [Message.system("sys"), Message.user_text("hi")]
        out = _serialize_messages(msgs)
        assert len(out) == 1
        assert out[0]["role"] == "user"

    def test_tool_result_as_user(self):
        from typoon.llm.anthropic import _serialize_messages
        msgs = [Message.tool_result_text("c1", "ok")]
        out = _serialize_messages(msgs)
        assert out[0]["role"] == "user"
        assert out[0]["content"][0]["type"] == "tool_result"
        assert out[0]["content"][0]["tool_use_id"] == "c1"

    def test_image_base64(self):
        from typoon.llm.anthropic import _serialize_parts
        parts = [ContentPart.of_image("data:image/jpeg;base64,/9j/abc")]
        out = _serialize_parts(parts)
        assert out[0]["type"] == "image"
        assert out[0]["source"]["media_type"] == "image/jpeg"
        assert out[0]["source"]["data"] == "/9j/abc"

    def test_tool_cache_breakpoint(self):
        from typoon.llm.anthropic import _serialize_tools
        tools = [ToolDef("a", "desc a", {"type": "object"}), ToolDef("b", "desc b", {"type": "object"})]
        out = _serialize_tools(tools)
        assert "cache_control" not in out[0]
        assert out[1]["cache_control"] == {"type": "ephemeral"}


class TestGemini:
    def test_system_instruction(self):
        from typoon.llm.gemini import _build_contents
        msgs = [Message.system("sys"), Message.user_text("hi")]
        system, contents = _build_contents(msgs)
        assert system == "sys"
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_assistant_tool_call(self):
        from typoon.llm.gemini import _build_contents
        tc = ToolCallMsg(id="c1", name="translate", arguments='{"x":1}')
        _, contents = _build_contents([Message.assistant(tool_calls=[tc])])
        assert contents[0].role == "model"
        assert contents[0].parts[0].function_call is not None
        assert contents[0].parts[0].function_call.name == "translate"

    def test_build_tools(self):
        from typoon.llm.gemini import _build_tools
        tools = [ToolDef("translate", "desc", {"type": "object", "properties": {}})]
        out = _build_tools(tools)
        assert len(out.function_declarations) == 1
        assert out.function_declarations[0].name == "translate"

    def test_find_tool_name(self):
        from typoon.llm.gemini import _find_tool_name
        tc = ToolCallMsg(id="c1", name="translate", arguments="{}")
        msgs = [Message.assistant(tool_calls=[tc])]
        assert _find_tool_name(msgs, "c1") == "translate"
        assert _find_tool_name(msgs, "c999") == "unknown"

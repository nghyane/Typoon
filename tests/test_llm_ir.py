"""Tests for LLM intermediate representation types."""

from typoon.llm.ir import ContentPart, Message, Role, ToolCallMsg, ToolResponse


class TestContentPart:
    def test_text(self):
        p = ContentPart.of_text("hello")
        assert p.text == "hello"
        assert p.image_data_uri is None

    def test_image(self):
        p = ContentPart.of_image("data:image/jpeg;base64,abc")
        assert p.image_data_uri == "data:image/jpeg;base64,abc"
        assert p.text is None


class TestMessage:
    def test_system(self):
        m = Message.system("hello")
        assert m.role == Role.SYSTEM
        assert m.text == "hello"

    def test_user_text(self):
        m = Message.user_text("hi")
        assert m.role == Role.USER
        assert m.text == "hi"
        assert len(m.parts) == 1

    def test_user_parts(self):
        m = Message.user_parts([
            ContentPart.of_text("look"),
            ContentPart.of_image("data:image/jpeg;base64,abc"),
        ])
        assert len(m.parts) == 2
        assert m.parts[0].text == "look"
        assert m.parts[1].image_data_uri == "data:image/jpeg;base64,abc"

    def test_assistant_with_tools(self):
        tc = ToolCallMsg(id="c1", name="translate", arguments='{"id":"b0"}')
        m = Message.assistant(text="thinking", tool_calls=[tc])
        assert m.role == Role.ASSISTANT
        assert m.text == "thinking"
        assert len(m.tool_calls) == 1

    def test_tool_result(self):
        m = Message.tool_result_text("c1", "ok")
        assert m.role == Role.TOOL_RESULT
        assert m.tool_call_id == "c1"
        assert m.text == "ok"


class TestToolResponse:
    def test_text_only(self):
        r = ToolResponse("ok")
        assert r.text == "ok"
        assert r.image_data_uri is None

    def test_with_image(self):
        r = ToolResponse("page 0", image_data_uri="data:image/jpeg;base64,xyz")
        assert r.text == "page 0"
        assert r.image_data_uri is not None

"""Original DeepSeek V4.1 template context and incremental DSML tool calls."""
import json
import re

from serve.frontend import ChatTemplate, OutputParser

DSML = "｜DSML｜"
OPEN = f"<{DSML}tool_calls>"
CLOSE = f"</{DSML}tool_calls>"


class DeepSeekTemplate(ChatTemplate):
    def __init__(self, path, tokenizer):
        token_id = tokenizer.special_ids.get("tokenizer.ggml.bos_token_id",
                                             tokenizer.ids.get("<｜begin▁of▁sentence｜>"))
        if token_id is None:
            raise ValueError("DeepSeek tokenizer is missing its BOS token")
        self.bos = tokenizer.decode([int(token_id)])
        super().__init__(path)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        kwargs.setdefault("bos_token", self.bos)
        kwargs.setdefault("enable_thinking", bool(kwargs.get("reasoning_effort")))
        wrapped = [{"type": "function", "function": t} if "function" not in t else t for t in tools or []]
        messages = [{**m, "content": "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))}
                    if isinstance(m.get("content"), list) else m for m in messages]
        return super().render(messages, tools=wrapped, add_generation_prompt=add_generation_prompt, **kwargs)

    def _detect_caps(self):
        caps = super()._detect_caps()
        tool = {"name": "strata_caps_call_0", "description": "strata_caps_description",
                "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}}
        user = {"role": "user", "content": "strata_caps_user"}
        prompt = self.render([user], [tool])
        caps["supports_tools"] = tool["name"] in prompt and OPEN in prompt
        for n, key in ((1, "supports_tool_calls"), (2, "supports_parallel_tool_calls")):
            calls = [{"function": {"name": tool["name"], "arguments": {"arg": str(i)}}} for i in range(n)]
            rendered = self.render([user, {"role": "assistant", "content": "", "tool_calls": calls}], [tool])
            caps[key] = rendered.count(f'<{DSML}invoke name="{tool["name"]}">') == n
        return caps


def forced_prefix(text):
    if not text:
        return text
    # Existing frontend tool_choice validation supplies these canonical forms.
    text = text.replace("<tool_call>", OPEN)
    text = re.sub(r"<function=([^>]+)>", lambda m: f'<{DSML}invoke name="{m[1]}">', text)
    return text.replace("<function=", f'<{DSML}invoke name="')


class DeepSeekOutputParser(OutputParser):
    """Translate only complete DSML tags; keep incomplete stream tags buffered.

    The existing parser owns reasoning, schema coercion, streamed JSON, and
    tool-call events. Text and quoted code retain their original spelling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dsml_pending = ""
        self.dsml_block = False
        self.dsml_invoke = False

    def feed(self, delta):
        self.dsml_pending += delta
        events = []
        prefixes = (f"<{DSML}", f"</{DSML}")
        while self.dsml_pending:
            at = self.dsml_pending.find("<")
            if at < 0:
                events += super().feed(self.dsml_pending); self.dsml_pending = ""; break
            if at:
                events += super().feed(self.dsml_pending[:at]); self.dsml_pending = self.dsml_pending[at:]
            value = self.dsml_pending
            if any(p.startswith(value) for p in prefixes):
                break
            if not value.startswith(prefixes):
                events += super().feed("<"); self.dsml_pending = value[1:]; continue
            end = value.find(">")
            if end < 0:
                break
            raw_tag = value[:end+1]; self.dsml_pending = value[end+1:]
            # V4.1 also emits a space after its DSML control token and the
            # shorter outer name "calls". Normalize structural tags only;
            # literal code and undeclared tools keep their original text.
            match = re.fullmatch(r'<(/?)｜DSML｜\s*([a-z_]+)(.*?)>', raw_tag, re.S)
            tag = raw_tag
            if match:
                name = "tool_calls" if match[2] == "calls" else match[2]
                tag = "<" + match[1] + DSML + name + match[3] + ">"
            converted = raw_tag
            if tag == OPEN and self.schemas and not self._in_code():
                self.dsml_block = True; converted = ""
            elif tag == CLOSE and self.dsml_block:
                self.dsml_block = False; converted = ""
            elif self.dsml_block:
                invoke = re.fullmatch(r'<｜DSML｜invoke name="([^"<>]+)">', tag)
                param = re.fullmatch(r'<｜DSML｜parameter name="([^"<>]+)"(?: string="(?:true|false)")?>', tag)
                if invoke:
                    self.dsml_invoke = True
                    converted = "<tool_call><function=" + invoke[1] + ">"
                elif tag == f"</{DSML}invoke>" and self.dsml_invoke:
                    self.dsml_invoke = False; converted = "</function></tool_call>"
                elif param and self.dsml_invoke:
                    converted = "<parameter=" + param[1] + ">"
                elif tag == f"</{DSML}parameter>" and self.dsml_invoke:
                    converted = "</parameter>"
            if converted:
                events += super().feed(converted)
        return events

    def finish(self, reason=None):
        events = super().feed(self.dsml_pending) if self.dsml_pending else []
        self.dsml_pending = ""
        return events + super().finish(reason)

"""DeepSeek's full-width control tokens and TP argument handoff (no model/GPU)."""
import unittest
import json
import tempfile
import threading
from types import SimpleNamespace
from pathlib import Path

from tools.strata_tokenizer import Tokenizer, BYTE_TO_UNICODE, SequentialSplitter, JOYAI_PATTERNS
from serve.server import Service, MockEngine, engine_args, openai_chunks, openai_collect
from serve.frontend import ChatTemplate, forced_call
from serve.deepseek41 import DeepSeekTemplate, DeepSeekOutputParser, forced_prefix


class DeepSeek41(unittest.TestCase):
    def tokenizer(self):
        pieces = list(BYTE_TO_UNICODE.values()) + ["<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>", "</think>"]
        return Tokenizer(pieces, [], [1]*256+[3,3,3,4], pre="joyai-llm",
                         special_ids={"tokenizer.ggml.eos_token_id":258})

    def test_full_width_special_roundtrip(self):
        tok = self.tokenizer()
        text = "<｜User｜>日本語 12345<｜Assistant｜></think>"
        self.assertEqual(tok.decode(tok.encode(text, parse_special=True)), text)
        self.assertEqual(tok.encode("<｜end▁of▁sentence｜>", parse_special=True), [258])

    def test_sequential_japanese_space_boundary(self):
        self.assertEqual(SequentialSplitter(JOYAI_PATTERNS).findall(" 日本語12345"), [" ", "日本語", "123", "45"])

    def test_eos_does_not_include_ordinary_punctuation(self):
        tok = self.tokenizer()
        template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        service = Service(SimpleNamespace(max_context=256), tok, template, model_name="deepseek-v4.1-flash-1017")
        self.assertEqual(service.stop_ids, {258})
        self.assertFalse(service.stop_ids.intersection(tok.encode("<>|endoftext")))

    def test_tp_does_not_add_pipeline_layer_split(self):
        args = ["generate", "--native", "/models/deepseek41.gguf", "--gpu", "0,1", "--tp", "2"]
        cfg = {"arch":"deepseek41", "exe":"strata", "args":args}
        self.assertNotIn("--layer-split", engine_args(cfg))
        self.assertIn("--tp", engine_args(cfg))

    def test_template_context_and_default_thinking(self):
        tok = self.tokenizer()
        tok.special_ids["tokenizer.ggml.bos_token_id"] = 256
        source = ('{{ bos_token }}{% for t in tools or [] %}{{ t.function.name }}{% endfor %}'
                  '{% for m in messages %}{{ m.content }}{% if m.tool_calls %}'
                  '{% set a = m.tool_calls[0].function.arguments %}{% if a is string %}{% set a = a | from_json %}{% endif %}'
                  '{{ a | tojson }}{% endif %}{% endfor %}'
                  '{% if enable_thinking %}<think>{% else %}</think>{% endif %}')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"template.jinja"; path.write_text(source)
            template = DeepSeekTemplate(path, tok)
            messages = [{"role":"user", "content":[{"type":"text", "text":"hi"}]},
                        {"role":"assistant", "content":"", "tool_calls":[{"function":{"name":"f", "arguments":'{"n":2}'}}]}]
            rendered = template.render(messages, tools=[{"name":"f"}])
            self.assertEqual(rendered, '<｜User｜>fhi{"n": 2}</think>')
            self.assertTrue(template.render([], reasoning_effort="high").endswith("<think>"))
            service = Service(SimpleNamespace(max_context=256), tok, template, model_name="deepseek-v4.1-flash-1017")
            for explicit in (False, True):
                kwargs = {}
                _, thinking, _ = service.prepare([{"role":"user", "content":"hi"}], None, kwargs, 8,
                    req={"chat_template_kwargs":{"enable_thinking":explicit}})
                self.assertEqual(thinking, explicit)

    def test_dsml_incremental_calls(self):
        tool = {"name":"lookup", "parameters":{"type":"object", "properties":{
            "text":{"type":"string"}, "n":{"type":"integer"}, "data":{"type":"object"}}}}
        text = ('Ready.\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="lookup">\n'
                '<｜DSML｜parameter name="text" string="true">日本語 a<b></｜DSML｜parameter>\n'
                '<｜DSML｜parameter name="n" string="false">23</｜DSML｜parameter>\n'
                '<｜DSML｜parameter name="data" string="false">{"x":[true,null]}</｜DSML｜parameter>\n'
                '</｜DSML｜invoke>\n<｜DSML｜invoke name="lookup">'
                '<｜DSML｜parameter name="text" string="true">second</｜DSML｜parameter>'
                '</｜DSML｜invoke>\n</｜DSML｜tool_calls>')
        for size in (1,3,len(text)):
            parser = DeepSeekOutputParser(thinking=False, tools=[tool], stream_tools=True)
            events = []
            for i in range(0,len(text),size): events += parser.feed(text[i:i+size])
            events += parser.finish("stop")
            calls = [e.call for e in events if e.kind == "tool_call"]
            self.assertEqual([c.name for c in calls], ["lookup", "lookup"])
            self.assertEqual(calls[0].arguments, {"text":"日本語 a<b>", "n":23, "data":{"x":[True,None]}})
            self.assertEqual(calls[1].arguments, {"text":"second"})
            for call in calls:
                pieces = [e.text for e in events if e.kind == "tool_args" and e.call.id == call.id]
                self.assertEqual(json.loads("".join(pieces)), call.arguments)
            content = "".join(e.text for e in events if e.kind == "content")
            self.assertIn("Ready.", content)
            self.assertNotIn("DSML", content)

    def test_dsml_literal_code_and_plain_text(self):
        text = '```xml\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="f"></｜DSML｜invoke>\n</｜DSML｜tool_calls>\n```'
        for tools in (None, [{"name":"f"}]):
            parser = DeepSeekOutputParser(thinking=False, tools=tools)
            events = []
            for ch in text: events += parser.feed(ch)
            events += parser.finish("stop")
            self.assertFalse(any(e.kind == "tool_call" for e in events))
            self.assertEqual("".join(e.text for e in events), text)

    def test_forced_tool_prefix_uses_original_dsml(self):
        self.assertEqual(forced_prefix('<tool_call>\n<function=lookup>\n'),
                         '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="lookup">\n')
        self.assertEqual(forced_prefix('<tool_call>\n<function='),
                         '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="')

    def test_forced_dsml_parameter_without_redundant_string_attribute(self):
        tool = {"name":"lookup", "parameters":{"type":"object", "properties":{"place":{"type":"string"}}}}
        parser = DeepSeekOutputParser(thinking=False, tools=[tool], stream_tools=True)
        events = parser.feed(forced_prefix('<tool_call>\n<function=lookup>\n'))
        suffix = '<｜DSML｜parameter name="place">Tokyo</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
        for ch in suffix: events += parser.feed(ch)
        events += parser.finish("stop")
        calls = [event.call for event in events if event.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {"place":"Tokyo"})

    def test_real_dsml_space_and_calls_variant(self):
        tool = {"name":"lookup", "parameters":{"type":"object", "properties":{"place":{"type":"string"}}}}
        suffix = ('<｜DSML｜ parameter name="place" string="true">Tokyo</｜DSML｜ parameter>\n'
                  '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        whole = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n' + suffix
        for size in (1,3,len(whole)):
            for forced in (False,True):
                parser = DeepSeekOutputParser(thinking=False, tools=[tool], stream_tools=True)
                events = parser.feed(forced_prefix('<tool_call>\n<function=lookup>\n')) if forced else []
                text = suffix if forced else whole
                for i in range(0,len(text),size): events += parser.feed(text[i:i+size])
                events += parser.finish("stop")
                calls = [event.call for event in events if event.kind == "tool_call"]
                self.assertEqual(len(calls),1)
                self.assertEqual(calls[0].arguments,{"place":"Tokyo"})
                pieces = [e.text for e in events if e.kind == "tool_args"]
                self.assertEqual(json.loads("".join(pieces)),calls[0].arguments)
        parser = DeepSeekOutputParser(thinking=False)
        events = parser.feed(whole) + parser.finish("stop")
        self.assertEqual("".join(e.text for e in events),whole)

    def test_forced_real_dsml_reaches_openai_response(self):
        tok = self.tokenizer(); tok.special_ids["tokenizer.ggml.bos_token_id"] = 256
        tool = {"name":"lookup", "parameters":{"type":"object", "properties":{"place":{"type":"string"}}}}
        suffix = ('<｜DSML｜ parameter name="place" string="true">Tokyo</｜DSML｜ parameter>\n'
                  '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        engine = MockEngine(tok,suffix,max_context=4096)
        engine.scripts = [tok.encode(suffix,parse_special=True)+[258]]; engine.script = engine.scripts[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"template.jinja"
            path.write_text('{{ bos_token }}<｜User｜>{{ messages[0].content }}<｜Assistant｜></think>')
            service = Service(engine,tok,DeepSeekTemplate(path,tok),model_name="deepseek-v4.1-flash-1017")
            service.model_arch = "deepseek41"
            force = forced_call({"type":"function", "function":{"name":"lookup"}},[tool])
            ids,thinking,maximum = service.prepare([{"role":"user", "content":"Tokyo"}],[tool],{},512,force=force)
            response = openai_collect(openai_chunks(service,{"temperature":0},ids,thinking,[tool],maximum,threading.Event(),force=force))
            choice = response["choices"][0]
            self.assertEqual(choice["finish_reason"],"tool_calls")
            call = choice["message"]["tool_calls"][0]
            self.assertEqual(call["function"]["name"],"lookup")
            self.assertEqual(json.loads(call["function"]["arguments"]),{"place":"Tokyo"})


if __name__ == "__main__":
    unittest.main()

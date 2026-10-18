#!/usr/bin/env python3
"""Smoke-test a running native Strata API without logging the API key."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = args.key_file.read_text().strip()
    if not key:
        parser.error("API key file is empty")
    result = {}

    def call(path, body=None, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(args.url.rstrip("/") + path,
                                         data=json.dumps(body).encode() if body is not None else None,
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
            return data if path == "/" or (body or {}).get("stream") else json.loads(data)

    try:
        call("/v1/models", auth=False)
        raise AssertionError("unauthenticated model request was accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 401, error.code
    result["unauthenticated_status"] = 401
    result["health"] = call("/health")
    models = call("/v1/models"); model = models["data"][0]["id"]
    assert "deepseek" in model.lower(), model
    result["model"] = model
    html = call("/").decode()
    assert "strata" in html.lower()
    result["ui"] = "Strata HTML served"
    base = {"model": model, "temperature": 0, "max_tokens": 32}
    for name, prompt in (("english", "Reply only Hello."), ("japanese", "日本語で一言、挨拶してください。")):
        response = call("/v1/chat/completions", {**base, "messages": [{"role": "user", "content": prompt}]})
        message = response["choices"][0]["message"]
        assert message.get("content"), message
        result[name] = message["content"]
    response = call("/v1/chat/completions", {**base, "stream": True,
                    "messages": [{"role": "user", "content": "Reply only streaming works."}]})
    assert b"data: [DONE]" in response and b'"content"' in response
    result["chat_sse"] = "PASS"
    response = call("/v1/responses", {"model": model, "input": "Reply only responses works.",
                    "temperature": 0, "max_output_tokens": 32})
    assert any(item.get("type") == "message" for item in response.get("output", [])), response
    result["responses"] = "PASS"
    response = call("/v1/messages", {**base, "messages": [{"role": "user", "content": "Reply only messages works."}]})
    assert any(item.get("type") == "text" and item.get("text") for item in response.get("content", [])), response
    result["anthropic"] = "PASS"
    tool = {"type": "function", "function": {"name": "lookup", "description": "Look up the given place.",
            "parameters": {"type": "object", "properties": {"place": {"type": "string"}}, "required": ["place"]}}}
    request = {**base, "max_tokens": 128, "tools": [tool],
               "tool_choice": {"type": "function", "function": {"name": "lookup"}},
               "messages": [{"role": "user", "content": "Call lookup with place Tokyo."}]}
    response = call("/v1/chat/completions", request)
    choice = response["choices"][0]; calls = choice["message"].get("tool_calls", [])
    assert choice["finish_reason"] == "tool_calls" and len(calls) == 1, response
    assert calls[0]["function"]["name"] == "lookup", calls
    assert json.loads(calls[0]["function"]["arguments"])["place"] == "Tokyo", calls
    result["tool_call"] = calls[0]["function"]
    messages = [*request["messages"], choice["message"], {"role": "tool", "tool_call_id": calls[0]["id"],
                "content": "Tokyo is a city in Japan. Reply briefly."}]
    response = call("/v1/chat/completions", {**base, "tools": [tool], "messages": messages})
    assert response["choices"][0]["message"].get("content"), response
    result["tool_history"] = "PASS"
    response = call("/v1/chat/completions", {**request, "stream": True})
    assert b'"tool_calls"' in response and b"data: [DONE]" in response
    result["tool_sse"] = "PASS"
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

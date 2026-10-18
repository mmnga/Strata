# Native DeepSeek V4.1

This opt-in Linux CUDA family runs DeepSeek-V4.1-Flash-1017 in Strata's own
inference driver. It reuses Strata's quantized matrix operators, grouped
experts, interleaved projections, expert cache and captured graphs, with
DeepSeek HC, Engram, compressed KV, sparse indexing and CPU experts.
It does not use a llama model, context, graph or inference scheduler.

The target retains its original Q4_K_M/Q6_K blocks. CPU experts are copied
byte for byte into pinned anonymous RAM; the original Engram tables remain
locked in RAM. Startup checks residency and refuses a paging fallback.
Dense weights live on the GPUs. No model conversion is required.

## Build

```sh
cmake -S . -B build/cuda -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_DEEPSEEK41=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_BUILD_TESTS=ON
cmake --build build/cuda --target strata -j24
```

GGML's CPU quantizer/dot-product helpers are used from that source tree.
The inference implementation is in `src/deepseek41/`. CUDA on Linux is the
validated backend; this family is not enabled on HIP or SYCL. Qwen remains
the default path when the DeepSeek build option is off.

## Run the adopted configuration

[The server example](../examples/deepseek41-server.json) contains the adopted
N21 settings. From the checkout root, copy it to an ignored local file:

```sh
cp examples/deepseek41-server.json strata-deepseek41.json
mkdir -p logs
```

Edit the target and DSpark model paths in that file. Keep the original model
shards together. Extract the tokenizer from the first target shard:

```sh
python tools/strata_tokenizer.py \
  --gguf /models/Q4_K_M/DeepSeek-V4.1-Flash-1017-Q4_K_M-00001-of-00010.gguf \
  --out pack/deepseek41/tokenizer --check
```

Start the standard Strata API and web app from the checkout root, using an
API key from a local file:

```sh
STRATA_API_KEY="$(cat /path/to/api-key.txt)" \
python serve/server.py --engine strata --config strata-deepseek41.json \
  --host 0.0.0.0 --port 8082
```

Use a nonempty key for LAN serving. API keys, exported tokenizers, model
files and generated local configurations are not part of the repository.
The example's relative paths are relative to the checkout root.

The tested profile uses TP2 across all 40 layers, context 262144 with F16
KV, 24 math participants (23 workers plus the caller), two-layer gate
lookahead and the original DSpark at draft depth 5. The CPU placement in
the example was measured on a Threadripper PRO 7965WX; adjust or remove
`STRATA_DS41_CPU_CORES` and `STRATA_DS41_ROUTER_CORE` on other machines.
The lookahead predicts placement hints; current-token routing still selects
the original top six experts.

DSpark retains its original MXFP4 expert blocks on the second GPU and uses
target features from layers 37/38/39. Its graph includes the original
five-position noise block, Markov chain and confidence threshold 0.75.
Greedy requests without active penalties use its proposals. The target
verifies them, keeps the accepted prefix and rolls back rejected KV and
Engram history. Other sampling uses ordinary native decoding.
`--no-dspark` disables the draft.

The adopted profile also enables attention graphs, the CPU expert flow,
selected cold-expert GPU streaming, packed FFN metadata, mapped CPU outputs,
grouped output projections, route handoff, shared softmax weights,
dimension-parallel attention values and prefetched query fragments.
Unadopted diagnostic paths remain disabled in the example.

## Validation and measured performance

The native tests cover metadata, HC, Engram hashing, routing, F16 ring and
sparse attention, YaRN, stable indexing, original MXFP4, CPU row phases,
cache leases, transfers, output projections and graph replay.
The original-model serial-versus-batch8 and rejected-tail checks passed
with bit-identical logits for N21. All six fixed-workload output sequences
and acceptance counts matched N19. Python tests cover DeepSeek's tokenizer,
BOS/EOS, template, DSML tools and tool history. API checks passed for English,
Japanese, chat streaming, Responses, Anthropic Messages, tool calls/history,
tool streaming, the web UI and authentication.

2026-10-18: Threadripper PRO 7965WX, 24 math participants, two RTX 5090 GPUs,
original Q4_K_M target, original corrected DSpark, TP2, context 262144/F16 KV.
Host model residency was 409.518 GiB, in addition to GPU-resident weights.

| Sequential workload | Output tokens | Decode tok/s |
| --- | ---: | ---: |
| Standard English | 1024 | 55.8836 |
| Standard Japanese | 1024 | 52.5986 |
| Held-out English | 1024 | 50.4281 |
| Held-out Japanese | 1024 | 55.8803 |
| Aggregate | 4096 | **53.5964** |

The aggregate is 4092 timed tokens divided by 76.3484 seconds of decode;
each request excludes its first output token from the timed count. It is
4.45% faster than N19 on this fixed workload, with the same output IDs and
acceptance counts. Model disk-read and major-fault deltas were zero, and
all serving processes had zero swap. The 70 tok/s target remains unmet.
These are workload-specific development measurements, not a speed guarantee
for other prompts or hardware. Raw runs and generated result cards stay local.

## Publishing an adopted improvement

After validation and adoption, commit the implementation, relevant tests,
portable example and documentation, then push `deepseek` to `mmnga/Strata`.
Enable the source-only push check once with
`git config core.hooksPath .githooks` if no other hook configuration is in use.
Run `python3 tools/deepseek41/check_push.py` before
`git push --set-upstream origin deepseek`. The check also runs as a pre-push
hook and inspects every outgoing commit, including files later deleted.

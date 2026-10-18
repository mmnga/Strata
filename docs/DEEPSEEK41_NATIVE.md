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
N40 settings. From the checkout root, copy it to an ignored local file:

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
dimension-parallel attention values and prefetched query fragments. N22 replaces DSpark's MXFP4 nibble table with an exactly
equivalent integer decode; original weight blocks and scales are retained.
N25 captures cached expert work, a system-acquire CPU completion wait, original
route-order combine and output transfer in one FFN graph. The empty-cache
variant also uses the completion wait. CPU/cold outputs are complete before
publication; exceptions abort outstanding waits. This requires native TP2,
route handoff, direct FFN, mirrored HC and `STRATA_DS41_TP_GRAPH=0`.
N26 distributes independent QK rows across more CTAs with a cooperatively
staged FP32 query. A separate pass uses the original full 256-thread
warp/shared max and sum reductions, sink and division, followed by the
unchanged scalar value FMA pass. No partial-softmax merge or new workspace.
N31 reuses the Qwen interleaved activation layout for exact Q4_K/Q6_K
projections with five to eight columns, at one fixed row per warp. Original
weight blocks, Q8 input bytes and per-column sums stay unchanged. The
DSpark IL buffer is prepared before graph capture. CHECK disables this
candidate throughout the reference prefix and serial steps. The example
sets `STRATA_DS41_WIDE_IL=1`; the general default stays off.
N32 stages two adjacent heads' probabilities and one shared KV row list,
then reuses each exact FP16 value read for two independent row-ordered FMA
recurrences. QK, full softmax and the existing persistent workspace stay
unchanged. `STRATA_DS41_VALUE_HEAD_PAIR=1` requires the score-grid path;
the default remains off and odd head counts are rejected only when enabled.
N40 reuses the accepted-window cache heat rule from the existing expert
path: credit each retained expert once per window, decay heat every four
windows, and track occurrence frequency separately with a 128-window decay.
Only observed occurrences train admissions; predictions remain placement
hints. Cache capacity, leases, pending-fill fences, replacement margin,
raw weight copies and target arithmetic stay unchanged.
`STRATA_DS41_WINDOW_HEAT=1` enables this rule; the general default stays off.
Unadopted diagnostic paths remain disabled in the example.

## Validation and measured performance

The native tests cover metadata, HC, Engram hashing, routing, F16 ring and
sparse attention, YaRN, stable indexing, original MXFP4, CPU row phases,
cache leases, transfers, output projections and graph replay.
The original-model serial-versus-batch8 and rejected-tail checks passed
with bit-identical logits for N32. All six fixed-workload output sequences
and acceptance counts for N32 matched N31. Python tests cover DeepSeek's tokenizer,
BOS/EOS, template, DSML tools and tool history. The N21 API checkpoint passed for English,
Japanese, chat streaming, Responses, Anthropic Messages, tool calls/history,
tool streaming, the web UI and authentication.

2026-10-18: Threadripper PRO 7965WX, 24 math participants, two RTX 5090 GPUs,
original Q4_K_M target, original corrected DSpark, TP2, context 262144/F16 KV.
Host model residency was 409.518 GiB, in addition to GPU-resident weights.

| Sequential workload | Output tokens | Decode tok/s |
| --- | ---: | ---: |
| Standard English | 1024 | 59.1552 |
| Standard Japanese | 1024 | 63.0427 |
| Held-out English | 1024 | 57.3685 |
| Held-out Japanese | 1024 | 63.6764 |
| Aggregate | 4096 | **60.6957** |

The aggregate is 4092 timed tokens divided by 67.4183 seconds of decode;
each request excludes its first output token from the timed count. It is
3.10% faster than adopted N32 on the same fixed requests/order.
Three long cases improved and standard English regressed. Cache placement
changes the existing CPU/GPU expert mix; DSpark counts differ, and output
IDs match N32 in 0 of six cases. This is an observed profile improvement,
not an isolated kernel timing or a statistical-confidence claim.
All six model disk-read and major-fault deltas were zero. The N40 live
benchmark/main-plus-two-locker snapshot had zero swap, with residency
enforcement enabled. Target math is unchanged from N32, so this metadata
change did not repeat its target CHECK. Its independent metadata oracle
passed 4000 accepted windows with 384 experts, repeated routes, retained
lengths 0..8, both decays, saturation and invalid-prefix immutability.
The 70 tok/s target remains unmet. The API checkpoint above remains N21;
these throughput results do not establish a later API checkpoint.
The MXFP4 unit gate passed all 4096 scale/nibble pairs against an independent
CPU exponent reference on each GPU, plus 48 bit-identical GU/Down graph
replays at H5120/F2304, top3, NT1..8 (1.04 seconds).
The N25 Host-wait/combined-output unit passed 144 exact mixed-source graph
replays and eight abort cases on both GPUs, including 32bit sequence rollover
and delayed CPU writes (0.62 seconds). The observed N26 aggregate gain
was consistent across all four long cases; statistical significance was not
established by this single changed-candidate suite.
The N26 score-grid unit passed 240 shape cells and 480 bit-identical graph
replays on both GPUs, including ring wrap, full-context positions, signed
zero and FP16 subnormals (1.38 seconds).
The wide-IL operator gate passed 48 shape cells and 96 bit-identical changed
graph replays on both GPUs, including five to eight columns, Q4_K/Q6_K,
zero rows, signed zeros, subnormal inputs and partial row groups. Q8 input
bytes also matched. These operators were unchanged by the N31 allocation
and original-prefix CHECK fixes; the real-model CHECK and fixed suite then
passed with the DSpark graph enabled.
The N32 adjacent-head value unit passed 240 shape cells and 480 bit-identical
graph replays on both GPUs (1.40 seconds), including the original whole-head
softmax and value FMA order, ring wrap, full-context positions, signed zeros
and FP16 subnormals.
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

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
N71 settings. From the checkout root, copy it to an ignored local file:

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
N49 ports ExpertPool's idle-worker clock cadence to the native CPU flow:
check the clock every 1024 pauses while continuing to check publication and
stop every pause. The 20 ms condition-variable deadline, participants,
affinity, task/completion protocol and arithmetic remain unchanged.
`STRATA_DS41_POOL_CLOCK=1` enables it and requires MWAITX disabled;
the default preserves the previous clock loop.
N55 omits the router and shared FFN projection copies on each layer's
non-owner rank with `STRATA_DS41_OWNER_WEIGHTS=1`. HC and its norms stay
mirrored. The original serial CHECK temporarily loads its oracle copies
and frees them before timing. This saved 558.691 MiB per GPU on the tested
model; cache capacity increased from 1193/814 to 1225/846 slots.
`STRATA_DS41_REGISTER_64M=1` uses 64 MiB host registration regions and the
same boundaries for GPU transfers. It avoids a large per-registration
kernel allocation that failed on the tested fragmented host. Every model
byte remains pinned and residency checks remain mandatory. Without this
flag the previous 1 GiB region size remains in use. Both region sizes
passed cross-boundary byte-copy tests on both GPUs.
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
| Standard English | 1024 | 59.1976 |
| Standard Japanese | 1024 | 63.1279 |
| Held-out English | 1024 | 57.3926 |
| Held-out Japanese | 1024 | 63.6879 |
| Aggregate | 4096 | **60.7360** |

N49's aggregate is 4092 timed tokens divided by 67.3736 seconds of decode;
each request excludes its first output token from the timed count. All four
long cases were slightly faster than N40 (60.6957 tok/s, 67.4183 seconds).
The difference is only 0.066%; this single changed-candidate suite does not
establish statistical significance, and the timing difference may be noise.
The profile is adopted for the small observed gain and verified reduction
of idle clock work. All six output sequences and DSpark counts matched N40.
Target and draft arithmetic, cache capacity and placement rules are unchanged.
All six disk-read and major-fault deltas were zero. The N49 live snapshot
of main plus two RAM lockers had zero swap, and 409.518 GiB residency was
enforced after every request. Actual-model batch8 and rejected-tail CHECK
had max_abs 0 and relative_rms 0 under the original 0.01 / 1e-4 limits.
Both original-clock and enabled-clock unit modes passed 7200 epochs and
266400 jobs with 1/4/24 participants, forced CV/default 20 ms waits,
3600 publish hooks, 36 intentional hook exceptions, late workers,
exactly-once callbacks/jobs, idle wake and shutdown (0.35 seconds enabled).

The included N40 heat rule measured 60.6957 tok/s, 3.10% over N32, with
three long cases faster and standard English slower. Its placement changes
affected the existing CPU/GPU expert mix: none of the six sequences or
DSpark counts matched N32. Target math was unchanged, so N40 did not
repeat target CHECK. Its independent metadata oracle passed 4000 accepted
windows with 384 experts, repeated routes, retained lengths 0..8,
both decays, saturation and invalid-prefix immutability.
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


2026-10-23, N55 on the same hardware and model settings: four long cases
measured 61.8837/65.4314 tok/s (standard EN/JA) and 55.4000/64.4661 tok/s
(held-out EN/JA). Aggregate: 4092 timed tokens / 66.5002 seconds =
**61.5336 tok/s**, an observed 1.313% over N49. All six generated sequences
and some draft acceptance counts changed, with identical prompt IDs;
held-out English was slower. This is a fixed-prompt throughput observation,
not an isolated same-output speedup or a significance claim. No arithmetic
or precision option was changed; larger caches can change CPU/GPU expert
assignment. Original batch8 and rollback CHECK passed at max_abs 0 and
relative_rms 0; that CHECK bypasses the expert cache, so it does not prove
cached-versus-CPU logit identity. Each case had zero model disk reads and
major faults; main plus two RAM lockers had zero swap at READY, and all
409.518 GiB remained resident. The first N55 startup failed during CUDA
host registration before READY; the 64 MiB option above allowed the full
run to complete. N55 is adopted locally; N49 is retained. No new API
checkpoint was run for N55; the 70 tok/s goal remains unmet.


N59 enables `STRATA_DS41_FFN_POST_FUSE=1` for the original direct FFN graph
with mirrored HC and CPU TP reduction. A separate CUDA operator combines
routed/shared outputs and updates the owner's four residual streams while
preserving the original route order, F32 intermediate rounding and HC FMA
order. The other rank receives the same FFN payload and performs its own
HC update. Original operators and the opt-out path are retained; CHECK uses
the original serial operators as its reference. All original CPU F16C
functions and numerical CUDA source remain unchanged.

2026-10-23 N59: 4092 timed tokens / 66.4788 seconds = **61.5535 tok/s**,
observed +0.0322% versus N55. Standard EN/JA:61.8710/65.4812; held-out
EN/JA:55.3853/64.5385 tok/s. This tiny single-suite difference can be noise;
it does not establish statistical significance. The local operator gate
was 1.14954x for combine plus HC, which does not predict whole-model gains.
The profile is adopted for its observed score and verified operator/launch
reduction. All six output sequences and draft counts matched N55. Actual
batch8/rollback CHECK max_abs and relative_rms were zero, model reads and
major faults were zero in all cases, and main plus two lockers had zero
swap at READY with all409.518GiB resident. Operator tests compared
44,236,800 values bitwise on both GPUs with NT1..8, capture on/off, mixed
CPU/GPU masks and in-place HC. No new N59 API checkpoint;70tok/s remains
unmet. N55 and earlier records remain available locally.


N60 enables `STRATA_DS41_HC_PRE_NORM=1` for H5120. The attention and FFN
HC pre-combinations and RMS normalization share one kernel. It retains the
original four FMA order, 256-thread sum reduction, runtime divisor and F32
multiplication order, and writes both original intermediate and normalized
outputs. The original serial CHECK disables this option for its reference.
Draft and output-head operators keep their previous paths.

N60 measured **61.7051 tok/s** (4092 timed tokens /66.3154s), observed
+0.2464% versus N59. Standard EN/JA:62.0026/65.6480; held-out EN/JA:
55.5106/64.7300 tok/s. This small single-suite gain does not establish
statistical significance. All six generated sequences and draft counts
matched N59. Actual batch8/rollback max_abs and relative_rms were zero;
all case model disk reads/major faults and READY3-process swap were zero,
with409.518GiB resident. Unit5,898,240 values matched bitwise on both GPUs,
NT1..8, captureon/off and changing signedzero/subnormal inputs. A local
operator gate measured1.05219x (15/16pairs faster), which is not a model
speed prediction. Original F16C functions and numerical operator sources
remain unchanged. No new N60 API checkpoint;70tok/s goal remains unmet.


N65 enables `STRATA_DS41_ROUTER_PIPE4=1` for the main router only. All40
source matrices must be BF16/type30 at5120x384; this is checked before RAM
registration. Their existing F32 expansion, per-thread FMA order and
256-thread reduction remain unchanged. Four immutable operand loads are
issued before their dependent FMAs. The original serial CHECK disables
this path for its reference. HC matrices are Q4_K and keep their original
quantized operators; draft and lookahead operators remain unchanged.

N65 measured **61.7552tok/s** (4092timed tokens/
66.2616s), observed+0.0812%vsN60.
Long cases:standard en 62.04362; standard ja 65.71045; heldout en 55.48956; heldout ja 64.87412. The tiny single-suite difference can be noise and is
not a significance claim. All6output sequences/draftcounts matchN60;
batch8/rollback CHECK max_abs/rRMS0, modelreads/majorfaults0 in allcases,
READYmain+2lockers swap0,409.518GiB resident. Routerunit221184values match
bitwise on2GPUs,NT1..8,captureon/off/changingweights/inputs/specialvalues.
A local router gate1.22122x is not a whole-model speed prediction.
CUDA test uploads explicitly complete on the consuming stream: pageable
host `cudaMemcpy` can return before final DMA completion. The existing
FFNpost and HCpre/RMS tests also use this ordering, with correctness-only
CTest runs. No production upload policy changed. No new N65API checkpoint;
server remains stopped and70tok/s remains unmet.


N69 adds `STRATA_DS41_MXFP4_PIPE4=1` to the adopted settings. It affects
only DSpark's original MXFP4 routed GU and Down kernels: preload four
successive 32-element blocks, then retain the original per-lane FMA and
warp reduction order. Original packed weights, integer ALU decode,
routes and clamps remain in use; no extra weight copy is allocated.
The default is off. The original main Q4_K_M kernels and CPU F16C code
are unchanged. The option requires 128-aligned input widths, checked
before draft graph capture (the model uses 5120 and 2304).

Two GPUs passed 4,810,752 full GU/Down bitwise comparisons at NT1..8
with changing inputs/routes and graph replay. The original decoder's
exhaustive exponent/nibble proof is reused. A complete GU+Down local
fixture measured 1.15148x across 16 balanced pairs; this is an operator
result, not a whole-model gain. The fixed four long requests measured
**62.3569 tok/s**, 4092 timed tokens / 65.6223 s, versus N65's 61.7552:
observed **+0.9742%**. All six generated ID sequences and draft counts
match N65. Long-case speeds were 62.5031 / 66.5461 tok/s (standard EN/JA)
and 55.9558 / 65.5769 (held-out EN/JA). Actual batch8/rollback CHECK
has max_abs=0 and relative_rms=0. Model reads and major faults were zero
for every case; all three processes had zero swap at READY. The original
409.518 GiB model residency checks passed. This single run does not
establish statistical significance. No new API checkpoint was run;
the server is stopped and the 70 tok/s goal remains unmet. Changes are
local; no push was performed.


N71 adds `STRATA_DS41_DSPARK_CONDITIONAL_HEAD=1`. It requires a CUDA 13.0+
build, graph replay and the existing positive confidence threshold. The
option defaults off; older CUDA builds retain the original path. On the
two RTX5090 GPUs, nested conditional graph capture passed every prefix
and threshold boundary. Full head comparisons preserve emitted token IDs,
confidence and logits bitwise; unreturned rows are skipped. The original
five-position noise block, all five Q6_K base logits, Markov matrices,
threshold0.75 and first-always-offer policy are retained. Conditional
nodes omit later Markov projections and sampling once that same policy
has stopped the prefix. Original target and CPU F16C arithmetic are unchanged.

The fixed long4 score was **62.5436 tok/s**, 4092 tokens /65.4264s, versus
N69's62.3569, observed **+0.2994%**. Both long English cases were slower;
both Japanese cases were faster. All six output ID sequences and draft
counts match N69. Batch8/rollback CHECK max_abs=0/relative_rms=0; all
inference disk reads and major faults were zero, READY's three processes
had zero swap, and409.518GiB residency passed. The local two-emitted-prefix
Markov-head fixture measured2.22529x (16/16 balanced pairs faster), which
is not a whole-model gain. The small observed long4 difference may be
noise; no statistical claim. No new API checkpoint was run; server stopped,
70tok/s unmet, local changes only and no push.

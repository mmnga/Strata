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
N91 settings. From the checkout root, copy it to an ignored local file:

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


N75 adds `STRATA_DS41_INDEX_RANK_BATCH=1`. Within the existing captured
attention body, up to eight token columns with at most 512 score rows use
one block per column in one kernel. CUB block radix ranking keeps the
original descending float score order and stable index ties, including
signed zero and NaN bit ordering. Larger row counts retain the original
per-column CUB device sort. Index projections, scoring, attention sums,
weights, scratch allocations and the graph shape keys are unchanged.
The option defaults off; the adopted example enables it.

On the same two RTX5090 GPUs and 24 CPU math participants, the fixed long4
aggregate was **62.8825 tok/s**, 4092 timed tokens / 65.0738 seconds, versus
N71's 62.5436: an observed **+0.5418%**. Standard English/Japanese measured
62.6907/67.4709 and held-out English/Japanese 56.1068/66.5998 tok/s. All four
were faster; all six generated ID sequences and DSpark counts matched N71.
This is one changed-candidate suite and does not establish significance.

The independent CUB ordering oracle matched 677376 IDs on both GPUs,
covering arbitrary float bits, NaNs, infinities, signed zeros, ties, token
counts 1..8 and row boundaries through 1024. The complete score/rank/
attention fixture matched 19660800 attention values bitwise plus 615796
ordered IDs, including paired checks. Its fixed NT3/ratio2/bucket512 local
gate was 1.50320x, 16/16 pairs faster, including scoring and attention and
excluding unchanged index-query projections. This is not a model gain.
Actual-model batch8/rollback CHECK had max_abs=0 and relative_rms=0;
all inference disk-read and major-fault deltas were zero, READY's three
processes had zero swap, and 409.518 GiB residency passed nine times.
Original Q4_K_M, CPU F16C math, DSpark MXFP4, TP2/all40, lookahead2 and
262144/F16 KV remain. No new API checkpoint; server stopped. The 70 tok/s
goal is unmet. Changes and commits remain local; no push.


N78 adds `STRATA_DS41_VALUE_ASYNC64=1`, requiring the adopted score-grid
and paired-head attention options. It keeps two heads, 64 dimensions and
64 threads per value CTA. Strata's asynchronous-copy commit/wait pattern
stages two 64-by-64 F16 tiles, loading the next tile while the original
row-ordered value FMAs consume the current tile. A wait and CTA barrier
protect every read and buffer reuse; the final tile waits for all copies.
The cache operator and raw F16 bits are retained. The QK and softmax
expressions are copied unchanged into the new CUDA file; none of the six
protected original CPU/CUDA math files changes. The extra 16 KiB is shared
memory per CTA; global workspace, weights, KV storage and cache slots stay
unchanged. The option defaults off; the adopted example enables it.

The fixed long4 aggregate was **63.6465 tok/s**, 4092 timed tokens /
64.2926 seconds, versus N75's 62.8825: observed **+1.2151%**. Standard
English/Japanese measured 63.1353/68.6154 and held-out English/Japanese
56.6223/67.6901 tok/s. All four were faster. Every prompt, generated ID
sequence and DSpark count matched N75 across all six requests. This one
changed-candidate suite does not establish statistical significance.

On both GPUs, 35389440 candidate attention values matched the original
full attention bitwise, including NT1..8, ratios0/1/2, ring wrap, full
context positions, changed graph inputs, signed zero and F16 subnormals.
The adopted paired baseline was checked on the same values. Another
786432 paired gate values matched. The fixed NT3/head32/D512/ratio2/
640-row complete QK/softmax/value fixture measured **1.39637x**, 16/16
pairs faster. It excludes projections and index ranking and is not a
whole-model gain. Actual-model batch8/rollback CHECK had max_abs=0 and
relative_rms=0; all inference disk-read and major-fault deltas were zero.
READY's three processes had zero swap, and 409.518 GiB residency passed
nine times. Cache slots remained1225/846, reserve1536MiB per GPU.
Original target Q4_K_M/F16C, DSpark MXFP4, TP2/all40,24math, lookahead2
and262144/F16KV remain. Server stopped; no new API checkpoint. The70tok/s
goal is unmet. All changes and commits remain local; no push.


N79 adds `STRATA_DS41_QUERY_REUSE=1`, requiring N78's asynchronous value
path. Each QK warp reuses its original 16 query fragments for more key
rows, retaining each key's 16 FMAs and shuffle reduction in their original
order. CUDA's one-time occupancy query reports 170 SMs and five resident
256-thread CTAs per SM on each tested GPU. The launch uses that 850-CTA
capacity divided across head/token pairs, capped by the original row grid.
No timing-selected grid or warp-count setting is used. The capability
query runs before capture; the host caches its result. Softmax, values,
global buffers, weights, KV storage and cache capacity are unchanged.
The option defaults off; capacity zero retains the original QK path.

The fixed long4 aggregate was **63.7420 tok/s**, 4092 timed tokens /
64.1963 seconds, versus N78's 63.6465: observed **+0.1500%**. Standard
English/Japanese measured63.2551/68.6988 and held-out English/Japanese
56.7039/67.7865 tok/s. All four were slightly faster; all six prompt/output
ID sequences and DSpark counts matched N78. This small difference may
include timing noise; one changed-candidate suite does not establish
statistical significance.

The two-GPU original/adopted/new attention oracle matched35389440 candidate
values bitwise, plus786432 paired gate values. It covers NT1..8,
ratios0/1/2, ring wrap, full context, changed graph inputs and F16 edges.
The complete attention local gate at NT3/head32/D512/ratio2/640rows was
1.11009x,16/16 pairs faster, excluding projections/index ranking. It is
not a whole-model speedup. Actual-model batch8/rollback CHECK had
max_abs=0 and relative_rms=0. Every inference disk-read and major-fault
delta was zero; READY's three processes had zero swap, and409.518GiB
residency passed nine times. Slots1225/846 and reserve1536MiB stayed fixed.
Original target Q4_K_M/F16C, DSpark MXFP4, TP2/all40,24math, lookahead2 and
262144/F16KV remain. Server stopped; no new API checkpoint.70tok/s remains
unmet. All changes and commits remain local; no push.


N80 adds `STRATA_DS41_NORM_Q8_FIRST=1`, requiring the adopted HC pre/RMS
path and the original Q4_K attention Q_A matrices5120x1280 with F32 norm
weights. HC collapse and the original256-thread RMS reduction first write
x and its inverse RMS. A parallel producer applies the same two rounded
F32 normalization multiplies and writes the original Q8_1 bytes, followed
immediately by the unchanged first Q_A projection. Both paths use three
kernels. This distributes norm application and avoids a normalized-input
reload; it does not reuse quantization across later projections, shared
experts or the draft. Inverse storage adds32 bytes per rank before capture.

The producer retains native_mmvq's per-file fast-math quantizer contract;
explicit non-FTZ `mul.rn.f32` instructions retain the original strict norm
multiplication order and subnormals. HC/RMS statistics stay in a separate
strict CUDA file. Other DeepSeek math files and CPU F16C are unchanged.
On the tested hardware the1280-row first projection uses plain Q8_1 at
NT1..4 and wide IL/CP8 at NT5..8. Serial CHECK clears the option throughout
its reference prefix. The option defaults off; the example enables it.

The fixed long4 aggregate was **64.2089 tok/s**,4092 timed tokens /
63.7295 seconds, versus N79's63.7420: observed **+0.7325%**. Standard
English/Japanese measured63.6574/69.2540 and held-out English/Japanese
57.0524/68.3984 tok/s. All four were faster; all six prompt/output ID
sequences and DSpark counts matched N79. One changed-candidate suite does
not establish statistical significance.

The two-GPU component oracle matched2511578 F32 values and3344640
plain/interleaved Q8_1 bytes exactly at NT1..8, with changed graph inputs,
zero blocks, signed zero, F32 subnormals and explicit null-weight captures.
The complete cold HC/pre/RMS/quant/Q_A gate used55 distinct matrices,
202752000 bytes, greater than twice reported L2100663296 bytes. Captured
weight pointers and every bank's outputs were checked. At fixed NT3, its
ratio was1.19372x,16/16 balanced pairs faster; this is not a model speedup.
Actual-model batch8/rollback CHECK had max_abs=0 and relative_rms=0. All
inference reads and major faults were zero, READY's three processes had
zero swap, and409.518GiB residency passed nine times. Cache slots stayed
1225/846 with1536MiB reserve per GPU. Original Q4_K_M, DSpark MXFP4,
TP2/all40,24math, lookahead2 and262144/F16KV remain. Server stopped;
no new API checkpoint;70tok/s remains unmet. No push.


N83 adds `STRATA_DS41_DSPARK_GROUPED_OA=1`. The original DSpark Q8_0
output-A has eight4096x1024 groups. Quantize the complete32768-wide input
once and use the existing exact grouped Qwen MMVQ body with Q8_0 traits.
This removes per-group gather/scatter and repeated quantization. Original
Q8_0 weights, dot products, reductions and output-B stay the same. The
ordinary Q8_0 dispatcher is unchanged. The option defaults off and guards
the original draft geometry; the example enables it. Pre-capture plain
quant storage grows by110592 bytes on the draft GPU, with no IL allocation.

The fixed four-long aggregate measured **64.5827 tok/s**,4092 timed tokens /
63.3606 seconds, versus N80's64.2089: observed **+0.5822%**. Standard
English/Japanese measured63.8808/69.8284 and held-out English/Japanese
57.3241/68.8875 tok/s. All four were faster. All six prompt/output ID
sequences and DSpark accepted/offered counts matched N80. A single changed
candidate suite does not establish statistical significance.

The two-GPU output-path oracle matched9953280 F32 values bit for bit
at NT1..8 across three changed captured inputs, including zero, signed
zero and subnormals. The fixed NT5 full inverse-RoPE/output-A/output-B
gate used three cold weight banks,240648192 bytes, greater than twice
reported L2100663296 bytes. All weight pointers and each bank's outputs
were checked. The ratio was1.46646x,16/16 balanced pairs faster, with
35->5 kernels; common input reset/submission/synchronization were included.
This is a component result. Actual-model batch8/rollback CHECK had max_abs=0
and relative_rms=0. All inference disk reads and major faults were zero,
READY's three processes had zero swap, and409.518GiB residency passed nine
times. Cache slots remained1225/846. Original main Q4_K_M/F16C, DSpark
MXFP4, TP2/all40,24math, lookahead2 and262144/F16KV remain. CUDA was tested.
Server stopped; no new API checkpoint.70tok/s remains unmet. No push.


N86 adds `STRATA_DS41_HC_COEFF_WARP=1`. The original HC coefficient
kernel uses one thread for its4x4 matrix. Sixteen warp lanes now compute
its independent entries. Row and column sums still add all four terms
in their original order, starting from the same zero or epsilon. The
original FMA, exponential, division, pre/post coefficients and twenty
Sinkhorn iterations remain. This applies to target and draft coefficients;
other math files and CPU F16C are unchanged. No new buffers or kernels are
added to the path. The option defaults off, requires four HC streams and
twenty iterations, and is enabled in the example. Serial CHECK disables
it throughout the original reference prefix.

The fixed four-long aggregate measured **65.3111 tok/s**,4092 timed tokens /
62.6540 seconds, versus N83's64.5827: observed **+1.1278%**. Standard
English/Japanese measured64.4065/70.8022 and held-out English/Japanese
57.8348/69.9081 tok/s. All four were faster. All six prompt/output ID
sequences and DSpark accepted/offered counts matched N83. The70tok/s
goal uses the four-long aggregate; it remains unmet. A single changed
candidate suite does not establish statistical significance.

The two-GPU oracle matched4447872 F32 values at NT1..8, with changed
full-HC captures and separate coefficient cases for zeros, saturation and
subnormals. The complete flat-RMS/Q8_1/Q4_K20480x24/HC coefficient gate
used729 cold matrices,201553920 bytes, greater than twice reported
L2100663296 bytes. Every captured weight pointer and each bank's outputs
were checked. At fixed NT3 both paths use four kernels; the ratio was
1.10017x,16/16 balanced pairs faster. This is a component result. Actual
batch8/rollback CHECK had max_abs=0 and relative_rms=0. All inference disk
reads and major faults were zero, READY's three processes had zero swap,
and409.518GiB residency passed nine times. Cache slots remained1225/846.
Original main Q4_K_M/F16C, DSpark MXFP4, TP2/all40,24math, lookahead2 and
262144/F16KV remain. Server stopped; no new API checkpoint. No push.


N88 adds `STRATA_DS41_HC_NORM_Q8=1` for the target's original Q4_K
HC projections20480x24. A strict256-thread kernel first computes the
original flat RMS reduction and runtime divisor. The existing N80 producer
then applies the same two rounded F32 multiplies with null gamma, writing
normalized F32 and Q8_1 together. The original HC projection consumes that
Q8_1 immediately. This avoids a normalized-input reload and distributes
normalization across the GPU. It adds32 bytes of inverse storage per rank
before capture. Both paths use four kernels, including N86 coefficients.

Only the producer's host width guard expands from5120 to5120-or20480;
its GPU body and explicit non-FTZ `mul.rn.f32` operations stay unchanged.
RMS statistics remain in a separate strict CUDA file. The24-row HC
projection uses plain input at NT1..4 and wide IL/CP8 at NT5..8. The option
defaults off, guards all original HC matrices, and is enabled in the example.
Serial CHECK disables it throughout the original reference prefix.
Shared experts, draft projections and the N80 first-Q_A path remain as before.

The fixed four-long aggregate measured **66.3757 tok/s**,4092 timed tokens /
61.6491 seconds, versus N86's65.3111: observed **+1.6300%**. Standard
English/Japanese measured65.2798/72.1367 and held-out English/Japanese
58.6549/71.2609 tok/s. All four were faster. All six prompt/output ID
sequences and DSpark accepted/offered counts matched N86. One changed
candidate suite does not establish statistical significance.

The two-GPU oracle matched4475320 F32 values and13378560 plain/IL bytes
at NT1..8, including changed captures, zero blocks, a whole-zero input,
signed zero, subnormals, null gamma and an observable original RMS inverse.
The full RMS/norm-Q8/Q4_K20480x24/N86-coefficient gate used729 cold matrices,
201553920 bytes, greater than twice L2100663296 bytes. All captured weight
pointers and each bank's outputs were checked. At fixed NT3, its ratio
was1.21240x,16/16 balanced pairs faster. This is a component result. Actual
batch8/rollback CHECK had max_abs=0 and relative_rms=0. All inference disk
reads and major faults were zero, READY's three processes had zero swap,
and409.518GiB residency passed nine times. Cache slots remained1225/846.
Original Q4_K_M/F16C, DSpark MXFP4, TP2/all40,24math, lookahead2 and
262144/F16KV remain.70tok/s is still unmet; server stopped, no new API
checkpoint. No push.


N89 adds `STRATA_DS41_HC_ROW_TOKENS=1`, requiring N88's HC input
producer. For the original target HC Q4_K20480x24 projections, each
row/token pair gets one CTA running the existing exact Qwen MMVQ body
with one row and one column. At three tokens this exposes72 independent
CTAs instead of12. Each dot retains its weight blocks, FMA order and
four-warp/warp reduction; this is not split-K. Ordinary multi-column
specializations compile the new offset condition away. The CUDA-only
adapter adds no buffers, weight copies or cache reservation.

The HC producer emits plain Q8_1 for all token counts on this path, so it
need not also write IL at NT5..8. Other projections retain their existing
selectors. Default is off; the example enables it. Serial CHECK clears
the flag throughout the original reference prefix.

The fixed four-long aggregate measured **67.0713 tok/s**,4092 timed tokens /
61.0097 seconds, versus N88's66.3757: observed **+1.0480%**. Standard
English/Japanese measured65.9183/73.0610 and held-out English/Japanese
59.0568/72.2141 tok/s. All four were faster. All six prompt/output ID
sequences and DSpark accepted/offered counts matched N88. One changed
candidate suite does not establish statistical significance.

The two-GPU oracle matched4475320 F32 values and5022720 plain Q8_1
bytes at NT1..8, with changed captures, zero blocks, a whole-zero input,
signed zero, subnormals, null gamma and an observable inverse. The
complete N88 HC preparation versus row/token path used729 cold matrices,
201553920 bytes, greater than twice L2100663296 bytes. Each captured
weight pointer and24-by-token128-thread grid, and every bank's outputs,
were checked. Both paths use four kernels; fixed NT3 measured1.18649x,
16/16 balanced pairs faster. This is a component result. Actual-model
batch8/rollback CHECK had max_abs=0 and relative_rms=0. All inference disk
reads and major faults were zero, READY's three processes had zero swap,
and409.518GiB residency passed nine times. Cache slots stayed1225/846.
Original Q4_K_M/F16C, DSpark MXFP4, TP2/all40,24math, lookahead2 and
262144/F16KV remain.70tok/s is still unmet; server stopped, no new API
checkpoint. No push.


N91 adds `STRATA_DS41_ROUTE_WARP=1` for target384/top6 and DSpark128/top3
routing. It reuses Qwen's register-resident candidate and stable warp
argmax structure. Each token uses one32-thread CTA instead of the
original256-thread selection with block barriers. DeepSeek retains its
original sqrt(softplus(logit)) probabilities, bias for ranking, ascending
expert ID on ties, unbiased selected probabilities and rank-ordered F32
weight sum, clamp and scale. Qwen's softmax and lane-ordered sum are not
used. The original route is unchanged by default. The example enables
the new path for both models; serial CHECK clears it for the reference.
No buffers, weight copies, cache reservations or placement policy change.

The fixed four-long aggregate measured **67.4701 tok/s**,4092 timed tokens /
60.6491 seconds, versus N89's67.0713: observed **+0.5946%**. Standard
English/Japanese measured66.4126/73.3907 and held-out English/Japanese
59.5179/72.4597 tok/s; all four were faster. All six prompt/output ID
sequences and DSpark accepted/offered counts matched N89. This is one
changed-candidate suite and does not establish statistical significance.

The two-GPU oracle matched2296080 F32 pipeline values and1296 route IDs
at NT1..8 with three changed captures. A further6480 F32 weights and6480
IDs matched for both route geometries across ten cases: bias, ties at
warp/bank boundaries, several winners in one lane, finite extremes,
underflow, subnormals, signed zero and a nonzero sum below the clamp.
The primary complete HCpre/RMS/N65 router/route gate used26 cold matrices,
204472320 bytes, greater than twice L2100663296 bytes. Every captured
matrix pointer, warp selector and bank's outputs were checked. Both paths
use three kernels. Fixed NT3 measured1.13469x,16/16 balanced pairs faster;
this is a component result, not the model's speed ratio.
Actual-model batch8/rollback CHECK had max_abs=0 and relative_rms=0. All
inference disk reads and major faults were zero, READY's three processes
had zero swap, and409.518GiB residency passed nine times. Cache slots
stayed1225/846. Original Q4_K_M/F16C, DSpark MXFP4, TP2/all40,24math,
lookahead2 and262144/F16KV remain.70tok/s is unmet; server stopped,
no new API checkpoint. Changes remain local; no push.

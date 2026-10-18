# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

## Publishing adopted DeepSeek changes

- The user's destination is `https://github.com/mmnga/Strata`, branch `deepseek`.
  After adopting and committing a validated improvement, push this branch to
  `origin` without asking again. Do not push a candidate before its adoption.
- Publish implementation, relevant tests, the portable server example and its
  documentation. Keep models, binaries, tokenizer exports, profiler output,
  runtime logs, experiment notebooks/images/prompts, trial configurations,
  API keys and machine-specific paths local. Preserve existing local records.
- Before pushing, fetch the remote branch and inspect every outgoing commit,
  including files later deleted. A cleanup commit does not remove a secret or
  generated artifact from earlier history. Keep local-only history in a local
  reference; do not publish it with `--all`, `--mirror`, tags or a force push.
- Enable the repository guard with `git config core.hooksPath .githooks` when
  there is no existing hook configuration to preserve. Run
  `python3 tools/deepseek41/check_push.py`, then
  `git push --set-upstream origin deepseek`. The guard checks the destination,
  outgoing paths and blobs; review its allowlist before extending the scope.
- Keep the deployed adopted server running while preparing a publication.
  Documentation and Git-only changes do not require another throughput run.

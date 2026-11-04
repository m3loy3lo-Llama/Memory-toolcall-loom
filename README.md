# ⚡ Local AI Agent Sanctuary

A clean, production-ready local AI engineering assistant capable of running any local foundation model (Gemma, Ministral, LFM, Llama, Qwen, etc. in GGUF format) with native filesystem, terminal, and semantic Headless IDE tool-calling, hierarchical sub-agent delegation, and tiered memory distillation.

---

## 🌟 Overview

This system provides **standard native tool calling** over a direct execution harness with dynamic model hot-swapping, hierarchical agent roles, and reasoning-centric memory compaction.

- **Foundational Engine**: `llama-server` (built for Apple Silicon / Metal & CUDA with `-ngl 99`, `--flash-attn auto`, and automatic multimodal `mmproj` detection).
- **Hierarchical Agent Architecture**:
  - **Lead Architect (`agent.js`)**: High-level reasoning, architectural strategy, heuristic evaluation, context compaction, and sub-agent coordination.
  - **Senior Developer (`subAgent.js`)**: Delegated worker agent (`delegate_subagent`) focused on surgical code navigation, symbol inspection, atomic diffs, and build verification.
- **Tiered Memory & Cold LoRA Distillation (`memoryManager.js`, `distillationEngine.js`)**:
  - **Hot Tier**: Active working memory tracking live file mutations and tool observations with 1280-dim semantic embeddings.
  - **Warm Tier**: Reasoning-centric compactions (`compact_context`) that capture the *why* (heuristics, invariants, decision paths) and establish **Context Checkpoints** to prune prior conversation history.
  - **Cold Tier**: Low-Rank Matrix Adapters (`ColdWeightAdapter`, $d=1280, r=16$) trained online across action transitions (`ActionChainTracker`) and swappable via contextual memory branches (`switch_memory_branch`).
- **Native Tools (`tools.js`)**:
  - `list_dir(path)`: Fast, filtered directory navigation.
  - `view_file(path, start_line, end_line)`: Line-numbered file inspection with range slicing.
  - `grep_search(query, path, is_regex)`: Fast symbol and pattern matching across project files.
  - `replace_file_content(path, target, replacement)`: Exact, atomic code replacement.
  - `write_file(path, content, overwrite)`: Safe file creation and updates.
  - `run_command(command)`: Sandboxed shell execution with stdout/stderr capture.
  - `generate_image(prompt, filename)`: On-demand visual generation with local SD-Turbo diffuser (`local_diffusion.py`).
  - `retrieve_memory`, `compact_context`, `delegate_subagent`, `switch_memory_branch`: Memory recall, checkpointing, and hierarchical delegation.
- **Headless IDE (`headless_ide/`)**: Semantic symbol extraction (`ide_get_symbols`), cross-project reference tracing (`ide_find_references`), context windows (`ide_get_context`), atomic unified diffs (`ide_apply_diff`, `ide_replace_all`), syntax linting (`ide_lint`), and build execution (`ide_build`).
- **Hot-Swapping**: Switch `.gguf` models on the fly through the Web UI dropdown or via `POST /api/models/switch`.
- **Frontend UI (`public/`)**: Responsive, dark-mode Sanctuary interface with live tool-activity cards, context checkpoint indicators, tiered memory inspector, and model selector.

---

## 📦 Dependencies & Prerequisites

### 1. `llama.cpp` (`llama-server`)
The backend manages `llama-server` automatically for local GPU-accelerated inference (Metal on Apple Silicon, CUDA on NVIDIA). Ensure `llama-server` is installed and available in your system `PATH`:

```bash
# macOS (via Homebrew)
brew install llama.cpp

# Verify installation
llama-server --version
```
*(Alternatively, download pre-built binaries or build from source at [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)).*

### 2. Gemma 4 GGUF Model (& Optional Vision Projector)
Place your quantized **Gemma 4 GGUF** model file in the project root directory (auto-detected on startup) or specify its location via `MODEL_PATH`:

- **Primary Model (Recommended)**: `gemma-4-12b-it-qat-q4_0.gguf`
- **Multimodal Vision Projector *(Optional)***: `mmproj-gemma-4-12b-it-qat-q4_0.gguf`
  - If placed in the same directory as the model, `modelManager.js` automatically pairs `mmproj-gemma-4-12b-it-qat-q4_0.gguf` to enable vision capabilities, and gracefully falls back to pure-text mode if omitted.
- *Other GGUF models (Ministral, Qwen, Llama, LFM, etc.) are also supported and can be hot-swapped at runtime via the Web UI.*

```bash
# Example: Download using Hugging Face CLI into the project root
huggingface-cli download google/gemma-4-12b-it-qat-q4_0-gguf \
  gemma-4-12b-it-qat-q4_0.gguf \
  mmproj-gemma-4-12b-it-qat-q4_0.gguf \
  --local-dir .
```

### 3. Node.js Runtime & NPM Packages
- **Node.js 18+** (tested on Node v20–v25)
- **NPM dependencies** (installed automatically via `npm install`):
  - `express` (`^4.18.2`) — HTTP server & REST API routing
  - `cors` (`^2.8.5`) — Cross-origin resource sharing
  - `diff` (`^9.0.0`) — Atomic unified diff parsing & patching for Headless IDE tools

### 4. Optional Dependencies
- **Local Image Generation (`generate_image` tool via `local_diffusion.py`)**:
  - Python 3.9+ with `torch`, `diffusers`, `transformers`, and `accelerate` (uses `stabilityai/sd-turbo` on MPS/CUDA/CPU):
    ```bash
    pip install torch diffusers transformers accelerate
    ```
- **C/C++ Linting & Formatting (`ide_lint` / `format` in Headless IDE)**:
  - `clang` and `clang-format` (included with Xcode Command Line Tools on macOS via `xcode-select --install`).

---

## 🚀 Quick Start

### 1. Install & Launch
```bash
npm install
npm start
# or ./boot.sh
```

Or specify a custom model path on launch:
```bash
MODEL_PATH="./gemma-4-12b-it-qat-q4_0.gguf" npm start
```

- **Web UI**: `http://localhost:3001`
- **Model API (`llama-server`)**: `http://127.0.0.1:8080`

### 2. Run Verification Suites
```bash
npm test
```

---

## 🏗️ Architecture

```text
.
├── server.js               # Express API server & endpoint routing
├── agent.js                # Lead Architect multi-turn loop & checkpoint pruning
├── subAgent.js             # Senior Developer delegated sub-agent execution loop
├── modelManager.js         # Background llama-server lifecycle & hot-swap manager
├── memoryManager.js        # Working memory & 1280-dim semantic embedding engine
├── distillationEngine.js   # Tiered compaction, ColdWeightAdapter (LoRA), & ActionChainTracker
├── tools.js                # OpenAI-format tool definitions & execution handlers
├── local_diffusion.py      # Local SD-Turbo image generation script (MPS/CUDA/CPU)
├── boot.sh                 # One-click startup script
├── headless_ide/           # Semantic Headless IDE microservice & tool integration
│   ├── server.js           # Standalone REST server for IDE operations
│   ├── vfs.js              # Virtual File System & context window provider
│   ├── symbolParser.js     # C/C++/JS lexical & syntactic symbol extractor
│   ├── manipulator.js      # Unified diff patching & regex manipulation
│   ├── feedback.js         # Compiler/syntax linting & build runner
│   ├── toolsIntegration.js # Direct tool bindings for agent loop
│   └── test_ide.js         # Headless IDE verification suite
├── public/                 # Sanctuary Dark Mode Web UI
│   ├── index.html          # Chat & memory dashboard interface
│   ├── style.css           # Glassmorphic dark styling & tool badges
│   └── app.js              # Reactive frontend with live tool & checkpoint cards
└── tests/                  # Automated test suites for distillation & checkpointing
    ├── test_distillation_and_hierarchy.js
    └── test_context_checkpoint.js
```

---

## ⚙️ Configuration & Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3001` | Port for the Express Web UI & Agent server |
| `MODEL_PATH` | Auto-detected `.gguf` | Path to the GGUF model file |
| `MMPROJ_PATH` | Auto-detected `mmproj-*.gguf` | Optional multimodal vision projector path |
| `LLAMA_HOST` | `127.0.0.1` | Host for `llama-server` |
| `LLAMA_PORT` | `8080` | Port for `llama-server` |
| `CTX_SIZE` | `0` (model max) | Context window size passed to `llama-server` |
| `GPU_LAYERS` | `99` | Number of layers offloaded to GPU (`-ngl`) |
| `WORKSPACE_DIR` | Project root | Root workspace directory for filesystem tools |
| `DIFFUSION_PYTHON_PATH` | `python3` | Python executable path for `local_diffusion.py` |

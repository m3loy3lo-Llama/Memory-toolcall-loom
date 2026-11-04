# Blueprint: Headless IDE for AI Assistant

## Overview
This project aims to create a "Headless IDE"—a specialized API-driven environment designed for an AI assistant to interact with a codebase with high precision, semantic awareness, and safety. Unlike a human-centric IDE, the priority here is **Semantic Density** (knowing what the code *means*) and **Surgical Manipulation** (making precise changes with minimal risk).

## Architecture Goals
- **Virtual File System (VFS):** The backend should maintain a representation of the project structure and file contents in memory (or a fast cache) to allow for rapid querying.
- **Semantic Mapping:** The IDE must parse C/C++ files to provide a map of symbols (functions, variables, classes) and their relationships (definitions, references).
- **Atomic Operations:** Edits should be performed as atomic units (e.g., "Replace all instances of X with Y" or "Apply this Diff") rather than line-by-line string replacements.
- **Feedback Loop:** The AI needs to be able to "see" the results of its actions (linting errors, build logs, test results) to self-correct.

## API Specification

### 1. Navigation & Knowledge (GET)
- `GET /project/structure`: Returns a JSON tree of the project.
    - *Requirement:* Identify file types (e.g., `.c`, `.h`, `.md`, `.json`).
- `GET /file/{path}/symbols`: Returns a list of all symbols in a file.
    - *Requirement:* Use `clang` or a similar parser to extract function names, variable names, and their line numbers.
- `GET /symbol/{path}/{symbol_name}/references`: Returns all locations where a specific symbol is used.
- `GET /file/{path}/context`: Returns a window of text around a specific line number.
- `GET /human/focus`: Returns the current file path and the line range currently highlighted/active in the human's UI.

### 2. Manipulation (POST)
- `POST /file/{path}/replace_all`: Performs a regex-based replacement across the entire file.
    - *Input:* `{ "regex": "...", "replacement": "..." }`
    - *Output:* `{ "success": boolean, "count": number, "diff": string }`
- `POST /file/{path}/apply_diff`: Applies a standard Unified Diff to a file.
    - *Input:* `{ "diff": "..." }`
    - *Output:* `{ "success": boolean, "error": string }`
- `POST /file/{path}/format`: Runs `clang-format` on the file and returns the formatted content.

### 3. Feedback & Execution (POST)
- `POST /project/lint`: Runs a linter (e.g., `cppcheck`) and returns a structured JSON of errors/warnings.
- `POST /project/build`: Triggers the build system (e.g., `make`) and returns the stdout/stderr.
- `POST /project/test`: Runs a specific test suite and returns the results.

## Technical Requirements
- **Runtime:** Node.js
- **Web Framework:** Express or Fastify
- **Parsing Engine:** `libclang` or a robust C/C++ parser to handle symbol extraction.
- **Formatting:** `clang-format`
- **Diffing:** A standard diff library (e.g., `diff` in npm).

## Implementation Roadmap for Builder AI
1. **Environment Setup:** Install Node.js, `clang`, `clang-format`, and `cppcheck`.
2. **VFS Implementation:** Create a service to load and cache the project structure and file contents.
3. **Symbol Parser:** Implement the logic to extract symbols from C/C++ files using `clang`.
4. **API Development:** Build the endpoints defined in the API Specification.
5. **Integration:** Connect the API to the existing `Gemma` folder's toolset.
6. **Testing:** Verify that the AI can successfully navigate to a symbol, identify its references, and apply a formatted change.

# 🧠 Headless IDE for AI Assistant

A specialized API-driven development environment built for AI assistants to navigate, inspect, and surgically manipulate codebases with semantic awareness.

Built based on the assistant specification in [`blueprint.md`](./blueprint.md).

---

## 🌟 Capabilities

1. **Virtual File System (VFS)**:
   * Fast in-memory hierarchy with detected language classifications (`.c`, `.h`, `.js`, `.ts`, `.md`, `.json`, etc.).
   * Code context windows around target line numbers with highlight markers.
   * Human focus state tracking (`/human/focus`).

2. **Semantic Symbol Extraction**:
   * Lexical & syntactic parser for C/C++ files:
     * Functions (definitions & declarations with return types and parameters).
     * Structs, enums, unions, and typedefs.
     * Macros (`#define FOO(...)`).
     * Global constants & variables.
   * Cross-project reference finder: locates all usage sites and distinguishes definitions from call sites.

3. **Atomic Code Manipulation**:
   * `replace_all`: Global regex replacements generating standard Unified Diffs.
   * `apply_diff`: Applies standard Unified Diff patches (`@@ ... @@`) safely and atomically using `diff`.
   * `format`: Code formatter (supports `clang-format` or whitespace normalization).

4. **Feedback & Execution**:
   * `lint`: Syntax checking using `clang -fsyntax-only` and `node -c`, returning structured line-by-line compiler diagnostics.
   * `build`: Triggers build systems (e.g. `make` or `npm run build`) and captures stdout, stderr, exit codes, and durations.
   * `test`: Executes test suites.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/project/structure?root=.&depth=4` | Returns JSON directory tree with file types & sizes |
| `GET` | `/file/:path/symbols` | Extracts all functions, structs, and symbols in a file |
| `GET` | `/symbol/:path/:symbol_name/references` | Finds all references and call sites of a symbol |
| `GET` | `/file/:path/context?line=N&window=10` | Returns lines around line `N` with focus indicators |
| `GET` | `/human/focus` | Returns active human file and line selection |
| `POST` | `/human/focus` | Updates active human file and line selection |
| `POST` | `/file/:path/replace_all` | Regex replacement across file + returns unified diff |
| `POST` | `/file/:path/apply_diff` | Applies a unified diff patch to a file atomically |
| `POST` | `/file/:path/format` | Formats code with clang-format or indenter |
| `POST` | `/project/lint` | Lints C/C++ or JS file and returns diagnostic errors |
| `POST` | `/project/build` | Triggers build command (`make`) and returns logs |
| `POST` | `/project/test` | Runs test command and returns results |

---

## 🛠️ Direct Assistant Tool Integration

All endpoints are also directly wired into the native toolset in `tools.js`:
* `ide_get_structure({ root, depth })`
* `ide_get_symbols({ path })`
* `ide_find_references({ symbol_name, target_dir })`
* `ide_get_context({ path, line, window })`
* `ide_apply_diff({ path, diff })`
* `ide_replace_all({ path, regex, replacement })`
* `ide_lint({ path })`
* `ide_build({ directory, command })`

---

## 🧪 Verification

Run the comprehensive 15-test suite:
```bash
node headless_ide/test_ide.js
```

Start the standalone microservice on port 3003:
```bash
node headless_ide/server.js
```

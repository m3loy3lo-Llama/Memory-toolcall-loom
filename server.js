import express from 'express';
import cors from 'cors';
import { vfs } from './vfs.js';
import { getFileSymbols, findSymbolReferences } from './symbolParser.js';
import { replaceAllInFile, applyDiffToFile, formatFile } from './manipulator.js';
import { lintFileOrProject, buildProject, testProject } from './feedback.js';

const app = express();
const PORT = process.env.HEADLESS_IDE_PORT || 3003;

app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '20mb' }));

// Helper to extract path from wildcard or query
function extractPath(req) {
  return req.params.path || req.params[0] || req.query.path || '';
}

// ---------------------------------------------------------------------------
// 1. Navigation & Knowledge (GET)
// ---------------------------------------------------------------------------

/**
 * GET /project/structure
 * Returns JSON tree of the project with language types and sizes
 */
app.get('/project/structure', async (req, res) => {
  try {
    const root = req.query.root || '.';
    const depth = Number(req.query.depth) || 6;
    const structure = await vfs.getProjectStructure(root, depth);
    res.json({ success: true, structure });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

/**
 * GET /file/:path/symbols
 * Returns a list of all symbols in a file (functions, structs, macros, variables)
 */
app.get(['/file/:path(*)/symbols', '/file/symbols'], async (req, res) => {
  try {
    const filePath = extractPath(req);
    if (!filePath) return res.status(400).json({ success: false, error: 'Path is required.' });

    const result = await getFileSymbols(filePath);
    res.json({ success: true, ...result });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

/**
 * GET /symbol/:path/:symbol_name/references
 * Returns all locations where a specific symbol is used
 */
app.get('/symbol/:path(*)/:symbol_name/references', async (req, res) => {
  try {
    const targetDir = req.params.path || '.';
    const symbolName = req.params.symbol_name;
    const references = await findSymbolReferences(symbolName, targetDir);
    res.json({ success: true, ...references });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

/**
 * GET /file/:path/context
 * Returns a window of text around a specific line number
 */
app.get(['/file/:path(*)/context', '/file/context'], async (req, res) => {
  try {
    const filePath = extractPath(req);
    const line = Number(req.query.line) || 1;
    const windowSize = Number(req.query.window) || 10;

    const context = await vfs.getFileContext(filePath, line, windowSize);
    res.json({ success: true, ...context });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

/**
 * GET /human/focus & POST /human/focus
 * Returns or sets the active file and line range highlighted in UI
 */
app.get('/human/focus', (req, res) => {
  res.json({ success: true, focus: vfs.getHumanFocus() });
});

app.post('/human/focus', (req, res) => {
  const { file, startLine, endLine } = req.body;
  if (!file) return res.status(400).json({ success: false, error: 'file is required.' });
  const focus = vfs.setHumanFocus(file, startLine, endLine);
  res.json({ success: true, focus });
});

// ---------------------------------------------------------------------------
// 2. Manipulation (POST)
// ---------------------------------------------------------------------------

/**
 * POST /file/:path/replace_all
 * Performs a regex-based replacement across the entire file and returns unified diff
 */
app.post(['/file/:path(*)/replace_all', '/file/replace_all'], async (req, res) => {
  try {
    const filePath = extractPath(req);
    const { regex, replacement, flags = 'g' } = req.body;

    if (!regex || replacement === undefined) {
      return res.status(400).json({ success: false, error: 'Parameters "regex" and "replacement" are required.' });
    }

    const result = await replaceAllInFile(filePath, regex, replacement, flags);
    res.json(result);
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

/**
 * POST /file/:path/apply_diff
 * Applies a standard Unified Diff to a file
 */
app.post(['/file/:path(*)/apply_diff', '/file/apply_diff'], async (req, res) => {
  try {
    const filePath = extractPath(req);
    const { diff } = req.body;

    if (!diff) {
      return res.status(400).json({ success: false, error: 'Parameter "diff" is required.' });
    }

    const result = await applyDiffToFile(filePath, diff);
    res.json(result);
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

/**
 * POST /file/:path/format
 * Formats file (clang-format or standard whitespace normalization)
 */
app.post(['/file/:path(*)/format', '/file/format'], async (req, res) => {
  try {
    const filePath = extractPath(req);
    const result = await formatFile(filePath);
    res.json(result);
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

// ---------------------------------------------------------------------------
// 3. Feedback & Execution (POST)
// ---------------------------------------------------------------------------

/**
 * POST /project/lint
 * Runs a linter / syntax check and returns structured errors/warnings
 */
app.post('/project/lint', async (req, res) => {
  try {
    const target = req.body.path || req.query.path || '.';
    const result = await lintFileOrProject(target);
    res.json(result);
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

/**
 * POST /project/build
 * Triggers the build system (e.g., make) and returns stdout/stderr
 */
app.post('/project/build', async (req, res) => {
  try {
    const { directory = '.', command = 'make' } = req.body;
    const result = await buildProject(directory, command);
    res.json(result);
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

/**
 * POST /project/test
 * Runs test suite and returns results
 */
app.post('/project/test', async (req, res) => {
  try {
    const { directory = '.', command = 'npm test' } = req.body;
    const result = await testProject(directory, command);
    res.json(result);
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

// ---------------------------------------------------------------------------
// Startup Helper
// ---------------------------------------------------------------------------

export function startHeadlessIDEServer(port = PORT) {
  return app.listen(port, () => {
    console.log(`\n⚡ Headless IDE microservice running at http://localhost:${port}`);
    console.log(`📡 Endpoints: /project/structure, /file/:path/symbols, /symbol/:path/:name/references`);
    console.log(`🛠️ Actions: /file/:path/apply_diff, /file/:path/replace_all, /project/lint, /project/build`);
  });
}

// If executed directly: `node server.js`
if (process.argv[1] && process.argv[1].endsWith('headless_ide/server.js')) {
  startHeadlessIDEServer();
}

export default app;

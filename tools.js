import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import { exec } from 'child_process';
import util from 'util';
import { fileURLToPath } from 'url';
import { IDE_TOOL_DEFINITIONS, IDE_TOOL_HANDLERS } from './headless_ide/toolsIntegration.js';

const execPromise = util.promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Workspace base directory (defaults to project root or WORKSPACE_DIR env var)
export const WORKSPACE_DIR = process.env.WORKSPACE_DIR
  ? path.resolve(process.env.WORKSPACE_DIR)
  : path.resolve(__dirname);

const IGNORED_NAMES = new Set([
  '.git', 'node_modules', '.DS_Store', '__pycache__', '.venv',
  '.idea', '.vscode', 'dist', 'build', 'cache'
]);

/**
 * Resolve a relative path against WORKSPACE_DIR and verify it stays inside or is accessible.
 */
export function resolveWorkspacePath(relPath = '.') {
  const normalized = path.normalize(relPath || '.');
  if (path.isAbsolute(normalized)) {
    return normalized;
  }
  const inWorkspace = path.resolve(WORKSPACE_DIR, normalized);
  if (fsSync.existsSync(inWorkspace)) {
    return inWorkspace;
  }
  const inCurrent = path.resolve(__dirname, normalized);
  if (fsSync.existsSync(inCurrent)) {
    return inCurrent;
  }
  return inWorkspace;
}

/**
 * Recursively list all files in the workspace, excluding ignored directories.
 * @returns {Promise<string[]>} Array of relative file paths.
 */
export async function listAllFiles() {
  const files = [];
  async function walk(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      const relPath = path.relative(WORKSPACE_DIR, fullPath);
      
      if (entry.isDirectory()) {
        if (!IGNORED_NAMES.has(entry.name)) {
          await walk(fullPath);
        }
      } else {
        files.push(relPath);
      }
    }
  }
  
  await walk(WORKSPACE_DIR);
  return files;
}



/**
 * Format bytes to readable size string
 */
function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Helper to test if a file buffer appears to be binary
 */
function isBinaryBuffer(buffer) {
  const checkLen = Math.min(buffer.length, 1024);
  for (let i = 0; i < checkLen; i++) {
    if (buffer[i] === 0) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Tool Implementations
// ---------------------------------------------------------------------------

/**
 * 1. list_dir
 */
export async function toolListDir({ path: relPath = '.' } = {}) {
  const targetDir = resolveWorkspacePath(relPath);
  try {
    const stats = await fs.stat(targetDir);
    if (!stats.isDirectory()) {
      return { error: `Path "${relPath}" is a file, not a directory.` };
    }

    const dirents = await fs.readdir(targetDir, { withFileTypes: true });
    const items = [];

    for (const d of dirents) {
      if (IGNORED_NAMES.has(d.name)) continue;

      const itemPath = path.join(targetDir, d.name);
      try {
        const itemStat = await fs.stat(itemPath);
        items.push({
          name: d.name,
          isDir: d.isDirectory(),
          size: d.isDirectory() ? '-' : formatSize(itemStat.size),
          bytes: d.isDirectory() ? 0 : itemStat.size
        });
      } catch {
        // Skip unreadable files
      }
    }

    // Sort directories first, then alphabetically
    items.sort((a, b) => {
      if (a.isDir && !b.isDir) return -1;
      if (!a.isDir && b.isDir) return 1;
      return a.name.localeCompare(b.name);
    });

    const displayPath = path.relative(WORKSPACE_DIR, targetDir) || '.';
    const lines = items.slice(0, 80).map(item => 
      `${item.isDir ? '📁 [DIR]' : '📄 [FILE]'} ${item.name} (${item.size})`
    );

    if (items.length > 80) {
      lines.push(`... and ${items.length - 80} more items`);
    }

    return {
      success: true,
      directory: displayPath,
      totalCount: items.length,
      listing: lines.join('\n'),
      entries: items
    };
  } catch (err) {
    return { error: `Failed to list directory "${relPath}": ${err.message}` };
  }
}

/**
 * 2. view_file
 */
export async function toolViewFile({ path: relPath, start_line, end_line } = {}) {
  if (!relPath) return { error: 'Parameter "path" is required for view_file.' };

  const filePath = resolveWorkspacePath(relPath);
  try {
    const stats = await fs.stat(filePath);
    if (stats.isDirectory()) {
      return { error: `Path "${relPath}" is a directory. Use list_dir instead.` };
    }

    if (stats.size > 2 * 1024 * 1024) {
      return { error: `File "${relPath}" is too large (${formatSize(stats.size)}). Max supported is 2MB.` };
    }

    const buffer = await fs.readFile(filePath);
    if (isBinaryBuffer(buffer)) {
      return { error: `File "${relPath}" appears to be binary. Cannot view as text.` };
    }

    const content = buffer.toString('utf-8');
    const allLines = content.split('\n');
    const totalLines = allLines.length;

    let start = 1;
    let end = totalLines;

    if (start_line !== undefined && !isNaN(start_line)) {
      start = Math.max(1, Number(start_line));
    }
    if (end_line !== undefined && !isNaN(end_line)) {
      end = Math.min(totalLines, Number(end_line));
    }

    if (start > end) {
      return { error: `Invalid range: start_line (${start}) > end_line (${end}).` };
    }

    const MAX_VIEW_LINES = 250;
    let truncated = false;
    if (end - start + 1 > MAX_VIEW_LINES) {
      end = start + MAX_VIEW_LINES - 1;
      truncated = true;
    }

    const sliced = allLines.slice(start - 1, end);
    const numbered = sliced.map((line, idx) => `${start + idx}: ${line}`).join('\n');

    return {
      success: true,
      path: relPath,
      totalLines,
      shownRange: `${start}-${end}`,
      content: numbered,
      truncated
    };
  } catch (err) {
    return { error: `Failed to view file "${relPath}": ${err.message}` };
  }
}

/**
 * 3. grep_search
 */
export async function toolGrepSearch({ query, path: relPath = '.', is_regex = false, case_sensitive = false } = {}) {
  if (!query) return { error: 'Parameter "query" is required for grep_search.' };

  const startDir = resolveWorkspacePath(relPath);
  let matcher;
  try {
    const flags = case_sensitive ? 'g' : 'gi';
    matcher = is_regex ? new RegExp(query, flags) : new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), flags);
  } catch (regErr) {
    return { error: `Invalid regex pattern: ${regErr.message}` };
  }

  const results = [];
  const MAX_RESULTS = 40;

  async function walk(currentDir, depth = 0) {
    if (depth > 6 || results.length >= MAX_RESULTS) return;

    let dirents;
    try {
      dirents = await fs.readdir(currentDir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const d of dirents) {
      if (results.length >= MAX_RESULTS) break;
      if (IGNORED_NAMES.has(d.name)) continue;

      const fullPath = path.join(currentDir, d.name);
      if (d.isDirectory()) {
        await walk(fullPath, depth + 1);
      } else if (d.isFile()) {
        try {
          const stat = await fs.stat(fullPath);
          if (stat.size > 500 * 1024) continue; // Skip files > 500KB for grep speed

          const buf = await fs.readFile(fullPath);
          if (isBinaryBuffer(buf)) continue;

          const text = buf.toString('utf-8');
          const lines = text.split('\n');
          const relFilePath = path.relative(WORKSPACE_DIR, fullPath);

          for (let i = 0; i < lines.length; i++) {
            if (results.length >= MAX_RESULTS) break;
            const line = lines[i];
            matcher.lastIndex = 0;
            if (matcher.test(line)) {
              results.push({
                file: relFilePath,
                line: i + 1,
                content: line.trim()
              });
            }
          }
        } catch {
          // Skip unreadable files
        }
      }
    }
  }

  try {
    await walk(startDir, 0);
    const summary = results.map(r => `${r.file}:${r.line}: ${r.content}`).join('\n');
    return {
      success: true,
      query,
      matchCount: results.length,
      capped: results.length >= MAX_RESULTS,
      results: results.length > 0 ? summary : 'No matches found.'
    };
  } catch (err) {
    return { error: `Grep search failed: ${err.message}` };
  }
}

/**
 * 4. replace_file_content
 */
export async function toolReplaceFileContent({ path: relPath, target, replacement } = {}) {
  if (!relPath || target === undefined || replacement === undefined) {
    return { error: 'Parameters "path", "target", and "replacement" are all required.' };
  }

  const filePath = resolveWorkspacePath(relPath);
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    const occurrences = content.split(target).length - 1;

    if (occurrences === 0) {
      return { error: `Target string was not found in file "${relPath}". Verify exact character sequence including spaces and newlines.` };
    }
    if (occurrences > 1) {
      return { error: `Target string appears ${occurrences} times in file "${relPath}". Provide more surrounding lines in "target" to uniquely identify the block.` };
    }

    const updated = content.replace(target, replacement);
    await fs.writeFile(filePath, updated, 'utf-8');

    return {
      success: true,
      path: relPath,
      message: `Successfully replaced content in ${relPath}.`
    };
  } catch (err) {
    return { error: `Failed to replace content in "${relPath}": ${err.message}` };
  }
}

/**
 * 5. write_file
 */
export async function toolWriteFile({ path: relPath, content, overwrite = false } = {}) {
  if (!relPath || content === undefined) {
    return { error: 'Parameters "path" and "content" are required.' };
  }

  const filePath = resolveWorkspacePath(relPath);
  try {
    const exists = fsSync.existsSync(filePath);
    if (exists && !overwrite) {
      return { error: `File "${relPath}" already exists. Set "overwrite": true if you explicitly wish to overwrite it.` };
    }

    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, content, 'utf-8');

    return {
      success: true,
      path: relPath,
      bytesWritten: Buffer.byteLength(content, 'utf-8'),
      action: exists ? 'overwritten' : 'created'
    };
  } catch (err) {
    return { error: `Failed to write file "${relPath}": ${err.message}` };
  }
}

/**
 * 6. run_command
 */
export async function toolRunCommand({ command, timeout_ms = 15000 } = {}) {
  if (!command) return { error: 'Parameter "command" is required.' };

  try {
    const { stdout, stderr } = await execPromise(command, {
      cwd: WORKSPACE_DIR,
      timeout: Math.min(60000, Number(timeout_ms) || 15000),
      maxBuffer: 512 * 1024
    });

    return {
      success: true,
      command,
      stdout: (stdout || '').trim(),
      stderr: (stderr || '').trim()
    };
  } catch (err) {
    return {
      success: false,
      command,
      exitCode: err.code || 1,
      stdout: (err.stdout || '').trim(),
      stderr: (err.stderr || err.message || '').trim()
    };
  }
}

/**
 * 7. generate_image
 */
export async function toolGenerateImage({ prompt, filename } = {}) {
  if (!prompt) return { error: 'Parameter "prompt" is required for generate_image.' };

  const PYTHON_PATH = process.env.DIFFUSION_PYTHON_PATH || 'python3';
  const SCRIPT_PATH = path.join(__dirname, 'local_diffusion.py');

  const safeName = filename
    ? filename.replace(/[^a-zA-Z0-9_-]/g, '_')
    : `generated_${Date.now()}`;
  const finalFilename = safeName.endsWith('.png') ? safeName : `${safeName}.png`;

  const outputDir = path.join(__dirname, 'public', 'images');
  await fs.mkdir(outputDir, { recursive: true });
  const outputPath = path.join(outputDir, finalFilename);

  const command = `"${PYTHON_PATH}" "${SCRIPT_PATH}" --prompt "${prompt.replace(/"/g, '\\"')}" --output "${outputPath}" --device mps`;

  try {
    await execPromise(command, { timeout: 45000 });
    const relativeUrl = `/images/${finalFilename}`;

    return {
      success: true,
      prompt,
      image_url: relativeUrl,
      savedPath: outputPath,
      message: `Image successfully generated: ${relativeUrl}`
    };
  } catch (err) {
    return {
      error: `Failed to generate image: ${err.message || err.stderr || err.stdout}`
    };
  }
}

import { memoryManager } from './memoryManager.js';
import { compactionManager } from './distillationEngine.js';
import { runSubAgent } from './subAgent.js';

export async function toolRetrieveMemory(args = {}) {
  return memoryManager.retrieveMemory(args);
}

export async function toolCompactContext({ reason, milestone, decision_path, heuristics = [] } = {}) {
  const result = await compactionManager.compactContext({
    reason: reason || 'Task milestone reached',
    milestone: milestone || 'Context compaction requested',
    decisionPath: decision_path || '',
    heuristics: Array.isArray(heuristics) ? heuristics : (heuristics ? [heuristics] : []),
    memoryManager
  });
  return {
    success: true,
    message: `Context compacted and distilled into cold LoRA weights for branch "${result.summary.branch}".`,
    compactionId: result.compactionId,
    milestone: result.summary.milestone,
    heuristicsLearned: result.summary.heuristics,
    distillationSteps: result.distillation.steps,
    handoverBrief: result.handoverBrief,
    checkpoint: result.checkpoint
  };
}

export async function toolDelegateSubagent({ task, instructions = '', target_files = [], max_steps = 8 } = {}) {
  if (!task) return { error: 'Task description is required for delegating to sub-agent.' };
  return await runSubAgent({
    task,
    instructions,
    targetFiles: target_files,
    maxSteps: Number(max_steps) || 8
  });
}

export async function toolSwitchMemoryBranch({ branch_name } = {}) {
  if (!branch_name) return { error: 'branch_name is required.' };
  return await compactionManager.switchBranch(branch_name);
}

// ---------------------------------------------------------------------------
// Tool Map and Definitions (Standard OpenAI Format)
// ---------------------------------------------------------------------------

export const TOOL_REGISTRY = {
  list_dir: toolListDir,
  view_file: toolViewFile,
  grep_search: toolGrepSearch,
  replace_file_content: toolReplaceFileContent,
  write_file: toolWriteFile,
  run_command: toolRunCommand,
  generate_image: toolGenerateImage,
  retrieve_memory: toolRetrieveMemory,
  compact_context: toolCompactContext,
  delegate_subagent: toolDelegateSubagent,
  switch_memory_branch: toolSwitchMemoryBranch,
  ...IDE_TOOL_HANDLERS
};

export const TOOL_DEFINITIONS = [
  {
    type: 'function',
    function: {
      name: 'list_dir',
      description: 'List files and directories in a given workspace path.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Relative path to directory. Defaults to "." (root).'
          }
        }
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'view_file',
      description: 'Read the contents of a text file with line numbers. Use start_line and end_line to inspect specific ranges.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Relative path to the file.'
          },
          start_line: {
            type: 'integer',
            description: 'Optional 1-based start line number.'
          },
          end_line: {
            type: 'integer',
            description: 'Optional 1-based end line number.'
          }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'grep_search',
      description: 'Search for text or regex pattern across workspace files. Returns line numbers and matching snippets.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The search string or regex pattern.'
          },
          path: {
            type: 'string',
            description: 'Optional directory path to restrict search within. Defaults to ".".'
          },
          is_regex: {
            type: 'boolean',
            description: 'Whether to interpret query as regular expression. Defaults to false.'
          }
        },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'replace_file_content',
      description: 'Replace an exact target block of code/text in a file with new replacement text.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Relative path to the file.'
          },
          target: {
            type: 'string',
            description: 'Exact string to be replaced, including whitespace and newlines.'
          },
          replacement: {
            type: 'string',
            description: 'New replacement text to insert.'
          }
        },
        required: ['path', 'target', 'replacement']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Create a new file or overwrite an existing file with content in the workspace.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Relative path to the target file.'
          },
          content: {
            type: 'string',
            description: 'The full string content to write.'
          },
          overwrite: {
            type: 'boolean',
            description: 'Set to true to allow overwriting an existing file. Defaults to false.'
          }
        },
        required: ['path', 'content']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'run_command',
      description: 'Execute a shell command inside the workspace directory.',
      parameters: {
        type: 'object',
        properties: {
          command: {
            type: 'string',
            description: 'The exact command line string to run.'
          }
        },
        required: ['command']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'generate_image',
      description: 'Generate a visual image based on a prompt using the local SD-Turbo diffusion model on GPU.',
      parameters: {
        type: 'object',
        properties: {
          prompt: {
            type: 'string',
            description: 'Descriptive visual prompt for the image.'
          },
          filename: {
            type: 'string',
            description: 'Optional output filename without extension.'
          }
        },
        required: ['prompt']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'retrieve_memory',
      description: 'Search and retrieve past tool observations, raw search outputs, and live file edit history with 1280-dim trainable latent embeddings.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Semantic search query to match against past observations or file edits.'
          },
          file: {
            type: 'string',
            description: 'Optional file path to retrieve all edits and observations related to that specific file.'
          },
          memory_id: {
            type: 'string',
            description: 'Optional exact memory ID for full raw output recall.'
          }
        }
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'compact_context',
      description: 'Compact conversation and working memory when a milestone is reached or context grows. Summarizes architectural reasoning (the why, heuristics, and decision paths), distills learnings into cold LoRA weights, and prepares a handover briefing.',
      parameters: {
        type: 'object',
        properties: {
          reason: {
            type: 'string',
            description: 'Why compaction was triggered (e.g. "Completed refactoring milestone", "Context limit approaching").'
          },
          milestone: {
            type: 'string',
            description: 'Summary of the achieved outcome or current workspace state.'
          },
          decision_path: {
            type: 'string',
            description: 'The reasoning and heuristics that governed decisions made so far.'
          },
          heuristics: {
            type: 'array',
            items: { type: 'string' },
            description: 'Key engineering rules, learned workspace patterns, or invariant constraints to preserve.'
          }
        },
        required: ['milestone']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'delegate_subagent',
      description: 'Delegate a concrete development or inspection sub-task to the Senior Developer sub-agent (hierarchical architecture). The sub-agent navigates data, executes atomic tools, and reports back.',
      parameters: {
        type: 'object',
        properties: {
          task: {
            type: 'string',
            description: 'Clear, specific task description for the sub-agent to execute.'
          },
          instructions: {
            type: 'string',
            description: 'Optional constraints or execution guidelines from the Lead Architect.'
          },
          target_files: {
            type: 'array',
            items: { type: 'string' },
            description: 'List of relevant files to inspect or modify.'
          },
          max_steps: {
            type: 'integer',
            description: 'Maximum tool steps for the sub-agent (default 8).'
          }
        },
        required: ['task']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'switch_memory_branch',
      description: 'Switch or create a project/task contextual memory branch to isolate task-specific weights while retaining global engineering heuristics.',
      parameters: {
        type: 'object',
        properties: {
          branch_name: {
            type: 'string',
            description: 'Name of the contextual branch (e.g. "auth_system", "c_firmware", "ui_refactor").'
          }
        },
        required: ['branch_name']
      }
    }
  },
  ...IDE_TOOL_DEFINITIONS
];

/**
 * Execute a tool by name with arguments
 */
export async function executeTool(name, args = {}) {
  const fn = TOOL_REGISTRY[name];
  if (!fn) {
    return { error: `Tool "${name}" is not recognized. Available tools: ${Object.keys(TOOL_REGISTRY).join(', ')}` };
  }
  try {
    return await fn(args);
  } catch (err) {
    return { error: `Exception executing tool ${name}: ${err.message}` };
  }
}

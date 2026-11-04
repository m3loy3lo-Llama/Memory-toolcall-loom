import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Workspace root (defaults to project root or WORKSPACE_DIR env var)
export const WORKSPACE_ROOT = process.env.WORKSPACE_DIR
  ? path.resolve(process.env.WORKSPACE_DIR)
  : path.resolve(__dirname, '..');

const IGNORE_PATTERNS = new Set([
  '.git', 'node_modules', '.DS_Store', '__pycache__', '.venv',
  '.idea', '.vscode', 'dist', 'build', 'cache', '.system_generated'
]);

const LANGUAGE_EXTENSIONS = {
  '.c': 'c',
  '.h': 'c_header',
  '.cpp': 'cpp',
  '.hpp': 'cpp_header',
  '.cc': 'cpp',
  '.js': 'javascript',
  '.mjs': 'javascript',
  '.ts': 'typescript',
  '.json': 'json',
  '.md': 'markdown',
  '.markdown': 'markdown',
  '.txt': 'text',
  '.py': 'python',
  '.html': 'html',
  '.css': 'css',
  '.sh': 'shell',
  '.ld': 'linker_script',
  '.s': 'assembly',
  '.asm': 'assembly',
  '.gba': 'gba_rom',
  '.elf': 'elf_binary'
};

// Global human focus state
let humanFocusState = {
  activeFile: null,
  startLine: 1,
  endLine: 1,
  updatedAt: new Date().toISOString()
};

export class VirtualFileSystem {
  constructor(rootDir = WORKSPACE_ROOT) {
    this.rootDir = rootDir;
  }

  resolvePath(relPath = '.') {
    const normalized = path.normalize(relPath || '.');
    if (path.isAbsolute(normalized)) {
      return normalized;
    }
    const inWorkspace = path.resolve(this.rootDir, normalized);
    if (fsSync.existsSync(inWorkspace)) return inWorkspace;

    // Check relative to Gemma root
    const inGemma = path.resolve(__dirname, '..', normalized);
    if (fsSync.existsSync(inGemma)) return inGemma;

    return inWorkspace;
  }

  /**
   * GET /project/structure
   * Returns a complete JSON tree of the project with language/type tags and file sizes.
   */
  async getProjectStructure(targetRelDir = '.', maxDepth = 6) {
    const rootPath = this.resolvePath(targetRelDir);
    if (!fsSync.existsSync(rootPath)) {
      throw new Error(`Target path "${targetRelDir}" does not exist.`);
    }

    const buildTree = async (currentPath, currentRel, depth) => {
      const stat = await fs.stat(currentPath);
      const name = path.basename(currentPath) || targetRelDir;

      if (!stat.isDirectory()) {
        const ext = path.extname(name).toLowerCase();
        return {
          name,
          path: currentRel,
          type: 'file',
          extension: ext || null,
          language: LANGUAGE_EXTENSIONS[ext] || 'unknown',
          sizeBytes: stat.size,
          modified: stat.mtime.toISOString()
        };
      }

      if (depth >= maxDepth) {
        return {
          name,
          path: currentRel,
          type: 'directory',
          truncated: true
        };
      }

      const entries = await fs.readdir(currentPath, { withFileTypes: true });
      const children = [];

      for (const entry of entries) {
        if (IGNORE_PATTERNS.has(entry.name)) continue;

        const childFull = path.join(currentPath, entry.name);
        const childRel = currentRel ? `${currentRel}/${entry.name}` : entry.name;

        try {
          const childNode = await buildTree(childFull, childRel, depth + 1);
          children.push(childNode);
        } catch {
          // Skip inaccessible entries
        }
      }

      // Sort dirs first, then files
      children.sort((a, b) => {
        if (a.type === 'directory' && b.type !== 'directory') return -1;
        if (a.type !== 'directory' && b.type === 'directory') return 1;
        return a.name.localeCompare(b.name);
      });

      return {
        name,
        path: currentRel || '.',
        type: 'directory',
        childCount: children.length,
        children
      };
    };

    return await buildTree(rootPath, targetRelDir === '.' ? '' : targetRelDir, 0);
  }

  /**
   * GET /file/{path}/context
   * Returns window of text around a line number.
   */
  async getFileContext(relPath, targetLine = 1, windowSize = 10) {
    const fullPath = this.resolvePath(relPath);
    if (!fsSync.existsSync(fullPath)) {
      throw new Error(`File "${relPath}" not found.`);
    }

    const content = await fs.readFile(fullPath, 'utf-8');
    const lines = content.split('\n');
    const totalLines = lines.length;

    const center = Math.max(1, Math.min(totalLines, Number(targetLine) || 1));
    const win = Math.max(1, Number(windowSize) || 10);

    const startLine = Math.max(1, center - win);
    const endLine = Math.min(totalLines, center + win);

    const contextLines = lines.slice(startLine - 1, endLine).map((text, idx) => ({
      line: startLine + idx,
      isTarget: startLine + idx === center,
      text
    }));

    return {
      path: relPath,
      targetLine: center,
      startLine,
      endLine,
      totalLines,
      lines: contextLines,
      formatted: contextLines.map(l => `${l.isTarget ? '👉 ' : '   '}${l.line}: ${l.text}`).join('\n')
    };
  }

  /**
   * GET /human/focus & POST /human/focus
   */
  getHumanFocus() {
    return { ...humanFocusState };
  }

  setHumanFocus(activeFile, startLine = 1, endLine = 1) {
    humanFocusState = {
      activeFile,
      startLine: Number(startLine) || 1,
      endLine: Number(endLine) || startLine || 1,
      updatedAt: new Date().toISOString()
    };
    return { ...humanFocusState };
  }
}

export const vfs = new VirtualFileSystem();

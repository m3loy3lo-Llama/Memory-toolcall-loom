import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import { vfs, WORKSPACE_ROOT } from './vfs.js';

/**
 * Robust C/C++ Symbol Parser
 */
export function parseCSymbols(content, relPath) {
  const symbols = [];
  const lines = content.split('\n');

  // Strip single-line and multi-line comments for cleaner matching while keeping line count intact
  const cleanLines = lines.map(line => {
    return line.replace(/\/\/.*$/, '');
  });

  // 1. Macros (#define NAME ...)
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    const macroMatch = line.match(/^#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)(?:\(([^)]*)\))?/);
    if (macroMatch) {
      symbols.push({
        name: macroMatch[1],
        kind: macroMatch[2] !== undefined ? 'macro_function' : 'macro_constant',
        line: i + 1,
        signature: macroMatch[0],
        file: relPath
      });
    }
  }

  // 2. Structs, Enums, Unions, Typedefs
  for (let i = 0; i < lines.length; i++) {
    const line = cleanLines[i].trim();

    // typedef struct/enum { ... } Name;
    const typedefMatch = line.match(/^typedef\s+(struct|enum|union)\s*(?:[A-Za-z_][A-Za-z0-9_]*)?\s*\{?/);
    if (typedefMatch) {
      // Look forward for closing brace and name
      let structName = null;
      let endLine = i + 1;
      for (let j = i; j < Math.min(lines.length, i + 50); j++) {
        const closeMatch = cleanLines[j].match(/\}\s*([A-Za-z_][A-Za-z0-9_]*)\s*;/);
        if (closeMatch) {
          structName = closeMatch[1];
          endLine = j + 1;
          break;
        }
      }
      if (structName) {
        symbols.push({
          name: structName,
          kind: `typedef_${typedefMatch[1]}`,
          line: i + 1,
          endLine,
          file: relPath
        });
      }
    } else {
      // struct Name { ... }
      const structDecl = line.match(/^(struct|enum|union|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{?/);
      if (structDecl) {
        symbols.push({
          name: structDecl[2],
          kind: structDecl[1],
          line: i + 1,
          file: relPath
        });
      }
    }
  }

  // 3. Function Definitions & Declarations
  // e.g., void AnimationInit(void) {
  // anim_instance_t* AnimCreateInstance(const anim_sequence_t* sequence, u8 slot) {
  const funcRegex = /^([A-Za-z_][A-Za-z0-9_*\s]+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*(\{)?/;
  const RESERVED_KEYWORDS = new Set([
    'if', 'else', 'while', 'for', 'switch', 'return', 'sizeof', 'case'
  ]);

  for (let i = 0; i < cleanLines.length; i++) {
    const line = cleanLines[i].trim();
    if (!line || line.startsWith('#') || line.startsWith('//') || line.startsWith('/*')) continue;

    const match = line.match(funcRegex);
    if (match) {
      const returnType = match[1].trim();
      const funcName = match[2].trim();
      const params = match[3].trim();
      const hasBrace = Boolean(match[4]);

      if (RESERVED_KEYWORDS.has(funcName)) continue;
      if (returnType.includes('typedef')) continue;

      // Find function end line if opening brace exists
      let endLine = i + 1;
      if (hasBrace || (i + 1 < cleanLines.length && cleanLines[i + 1].trim().startsWith('{'))) {
        let braceCount = 0;
        let started = false;
        for (let j = i; j < Math.min(cleanLines.length, i + 500); j++) {
          for (const char of cleanLines[j]) {
            if (char === '{') {
              braceCount++;
              started = true;
            } else if (char === '}') {
              braceCount--;
              if (started && braceCount === 0) {
                endLine = j + 1;
                break;
              }
            }
          }
          if (started && braceCount === 0) break;
        }
      }

      symbols.push({
        name: funcName,
        kind: hasBrace ? 'function_definition' : 'function_declaration',
        returnType,
        parameters: params,
        line: i + 1,
        endLine: endLine > i + 1 ? endLine : i + 1,
        signature: `${returnType} ${funcName}(${params})`,
        file: relPath
      });
    }
  }

  // 4. Global Variables & Constants (e.g., const anim_sequence_t ANIM_APPLE_BREATH = ...)
  for (let i = 0; i < cleanLines.length; i++) {
    const line = cleanLines[i].trim();
    const varMatch = line.match(/^(?:static\s+|const\s+|extern\s+)*([A-Za-z_][A-Za-z0-9_*\s]+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*=\s*[\{\[\w]/);
    if (varMatch) {
      const varType = varMatch[1].trim();
      const varName = varMatch[2].trim();
      if (!RESERVED_KEYWORDS.has(varName) && !symbols.some(s => s.name === varName && s.line === i + 1)) {
        symbols.push({
          name: varName,
          kind: 'global_variable',
          type: varType,
          line: i + 1,
          file: relPath
        });
      }
    }
  }

  // Deduplicate and sort by line number
  const seen = new Set();
  const deduped = [];
  for (const sym of symbols) {
    const key = `${sym.name}:${sym.line}`;
    if (!seen.has(key)) {
      seen.add(key);
      deduped.push(sym);
    }
  }

  deduped.sort((a, b) => a.line - b.line);
  return deduped;
}

/**
 * JavaScript / TypeScript Symbol Parser
 */
export function parseJSSymbols(content, relPath) {
  const symbols = [];
  const lines = content.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // function foo(...)
    const fnMatch = line.match(/^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(([^)]*)\)/);
    if (fnMatch) {
      symbols.push({
        name: fnMatch[1],
        kind: 'function',
        parameters: fnMatch[2],
        line: i + 1,
        file: relPath
      });
      continue;
    }

    // const/let foo = (...) => or function
    const arrowMatch = line.match(/^(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?(?:\(([^)]*)\)|[A-Za-z0-9_$]+)\s*=>/);
    if (arrowMatch) {
      symbols.push({
        name: arrowMatch[1],
        kind: 'arrow_function',
        parameters: arrowMatch[2] || '',
        line: i + 1,
        file: relPath
      });
      continue;
    }

    // class Foo
    const classMatch = line.match(/^(?:export\s+(?:default\s+)?)?class\s+([A-Za-z0-9_$]+)/);
    if (classMatch) {
      symbols.push({
        name: classMatch[1],
        kind: 'class',
        line: i + 1,
        file: relPath
      });
      continue;
    }

    // export const FOO = ...
    const constMatch = line.match(/^(?:export\s+)?const\s+([A-Za-z0-9_$]+)\s*=/);
    if (constMatch) {
      symbols.push({
        name: constMatch[1],
        kind: 'constant',
        line: i + 1,
        file: relPath
      });
    }
  }

  return symbols;
}

/**
 * Universal Symbol Extraction Entrypoint
 */
export async function getFileSymbols(relPath) {
  const fullPath = vfs.resolvePath(relPath);
  if (!fsSync.existsSync(fullPath)) {
    throw new Error(`File "${relPath}" not found.`);
  }

  const content = await fs.readFile(fullPath, 'utf-8');
  const ext = path.extname(relPath).toLowerCase();

  let symbols = [];
  if (['.c', '.h', '.cpp', '.hpp', '.cc'].includes(ext)) {
    symbols = parseCSymbols(content, relPath);
  } else if (['.js', '.mjs', '.ts', '.jsx', '.tsx'].includes(ext)) {
    symbols = parseJSSymbols(content, relPath);
  }

  return {
    path: relPath,
    totalSymbols: symbols.length,
    symbols
  };
}

/**
 * Cross-project Symbol Reference Search
 */
export async function findSymbolReferences(symbolName, targetDir = '.') {
  if (!symbolName) throw new Error('Symbol name is required.');

  const rootPath = vfs.resolvePath(targetDir);
  const regex = new RegExp(`\\b${symbolName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g');
  const results = [];

  const walk = async (currentDir, depth = 0) => {
    if (depth > 6) return;
    const entries = await fs.readdir(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      if (['.git', 'node_modules', '.DS_Store', 'dist', 'build'].includes(entry.name)) continue;

      const full = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        await walk(full, depth + 1);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (!['.c', '.h', '.cpp', '.hpp', '.js', '.ts', '.json', '.md'].includes(ext)) continue;

        try {
          const content = await fs.readFile(full, 'utf-8');
          if (!regex.test(content)) continue;

          const rel = path.relative(WORKSPACE_ROOT, full);
          const lines = content.split('\n');

          for (let i = 0; i < lines.length; i++) {
            const lineText = lines[i];
            regex.lastIndex = 0;
            if (regex.test(lineText)) {
              const isDef = /(?:void|int|u8|u16|u32|const|static|struct|class|function)\s+/.test(lineText) &&
                            (lineText.includes('(') || lineText.includes('=') || lineText.includes('{'));

              results.push({
                file: rel,
                line: i + 1,
                isLikelyDefinition: isDef,
                snippet: lineText.trim()
              });
            }
          }
        } catch {
          // Skip unreadable files
        }
      }
    }
  };

  await walk(rootPath);

  return {
    symbol: symbolName,
    referenceCount: results.length,
    references: results
  };
}

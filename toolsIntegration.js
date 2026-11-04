import { vfs } from './vfs.js';
import { getFileSymbols, findSymbolReferences } from './symbolParser.js';
import { replaceAllInFile, applyDiffToFile, formatFile } from './manipulator.js';
import { lintFileOrProject, buildProject } from './feedback.js';

export const IDE_TOOL_DEFINITIONS = [
  {
    type: 'function',
    function: {
      name: 'ide_get_structure',
      description: 'Get project structure tree with detected language types and file sizes.',
      parameters: {
        type: 'object',
        properties: {
          root: {
            type: 'string',
            description: 'Subdirectory to map. Defaults to "."'
          },
          depth: {
            type: 'integer',
            description: 'Max directory traversal depth (default 4).'
          }
        }
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'ide_get_symbols',
      description: 'Parse a C, C++, or JS file to extract all symbols (functions, structs, typedefs, macros, variables) with line numbers.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Relative path to the source file (e.g. src/index.js or src/main.c)'
          }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'ide_find_references',
      description: 'Search the codebase for all occurrences and definitions of a symbol name.',
      parameters: {
        type: 'object',
        properties: {
          symbol_name: {
            type: 'string',
            description: 'The exact symbol/function/struct name to search for.'
          },
          target_dir: {
            type: 'string',
            description: 'Directory to restrict search to. Defaults to "."'
          }
        },
        required: ['symbol_name']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'ide_get_context',
      description: 'Get a window of lines around a specific line number with focus indicators.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Relative path to file.'
          },
          line: {
            type: 'integer',
            description: 'Target line number.'
          },
          window: {
            type: 'integer',
            description: 'Number of lines before and after to include (default 10).'
          }
        },
        required: ['path', 'line']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'ide_apply_diff',
      description: 'Apply a standard unified diff patch to a file atomically.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Relative path to the file to patch.'
          },
          diff: {
            type: 'string',
            description: 'Unified diff text containing @@ ... @@ hunks.'
          }
        },
        required: ['path', 'diff']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'ide_replace_all',
      description: 'Perform regex-based replacement across a file and return a unified diff of the changes.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Relative path to the file.'
          },
          regex: {
            type: 'string',
            description: 'Regex pattern to match.'
          },
          replacement: {
            type: 'string',
            description: 'Replacement text.'
          }
        },
        required: ['path', 'regex', 'replacement']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'ide_lint',
      description: 'Check a C/C++ or JS file for syntax and compiler errors.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Relative path to file to lint.'
          }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'ide_build',
      description: 'Execute build command (e.g. make) in a project directory and capture compiler output and timing.',
      parameters: {
        type: 'object',
        properties: {
          directory: {
            type: 'string',
            description: 'Directory where build command is run. Defaults to "."'
          },
          command: {
            type: 'string',
            description: 'Build command to execute. Defaults to "make".'
          }
        }
      }
    }
  }
];

export const IDE_TOOL_HANDLERS = {
  ide_get_structure: async ({ root = '.', depth = 4 } = {}) => {
    return await vfs.getProjectStructure(root, Number(depth) || 4);
  },
  ide_get_symbols: async ({ path: relPath } = {}) => {
    return await getFileSymbols(relPath);
  },
  ide_find_references: async ({ symbol_name, target_dir = '.' } = {}) => {
    return await findSymbolReferences(symbol_name, target_dir);
  },
  ide_get_context: async ({ path: relPath, line = 1, window = 10 } = {}) => {
    return await vfs.getFileContext(relPath, Number(line) || 1, Number(window) || 10);
  },
  ide_apply_diff: async ({ path: relPath, diff } = {}) => {
    return await applyDiffToFile(relPath, diff);
  },
  ide_replace_all: async ({ path: relPath, regex, replacement } = {}) => {
    return await replaceAllInFile(relPath, regex, replacement);
  },
  ide_lint: async ({ path: relPath } = {}) => {
    return await lintFileOrProject(relPath);
  },
  ide_build: async ({ directory = '.', command = 'make' } = {}) => {
    return await buildProject(directory, command);
  }
};

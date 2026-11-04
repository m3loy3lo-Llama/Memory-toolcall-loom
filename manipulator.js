import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import * as Diff from 'diff';
import { exec } from 'child_process';
import util from 'util';
import { vfs } from './vfs.js';

const execPromise = util.promisify(exec);

/**
 * Atomic Regex Replacement across a file with unified diff generation
 */
export async function replaceAllInFile(relPath, regexStr, replacement, flags = 'g') {
  const fullPath = vfs.resolvePath(relPath);
  if (!fsSync.existsSync(fullPath)) {
    throw new Error(`File "${relPath}" not found.`);
  }

  const originalContent = await fs.readFile(fullPath, 'utf-8');
  let regex;
  try {
    regex = new RegExp(regexStr, flags);
  } catch (err) {
    throw new Error(`Invalid regex: ${err.message}`);
  }

  const matches = originalContent.match(regex);
  const matchCount = matches ? matches.length : 0;

  if (matchCount === 0) {
    return {
      success: true,
      count: 0,
      diff: '',
      message: 'No matches found. File content unchanged.'
    };
  }

  const newContent = originalContent.replace(regex, replacement);

  // Generate unified diff
  const patch = Diff.createTwoFilesPatch(
    `a/${relPath}`,
    `b/${relPath}`,
    originalContent,
    newContent,
    'original',
    'modified'
  );

  // Write new content atomically
  await fs.writeFile(fullPath, newContent, 'utf-8');

  return {
    success: true,
    path: relPath,
    count: matchCount,
    diff: patch
  };
}

/**
 * Apply a standard Unified Diff to a target file
 */
export async function applyDiffToFile(relPath, diffText) {
  const fullPath = vfs.resolvePath(relPath);
  if (!fsSync.existsSync(fullPath)) {
    throw new Error(`File "${relPath}" not found.`);
  }

  if (!diffText || typeof diffText !== 'string') {
    throw new Error('Diff text is required.');
  }

  const originalContent = await fs.readFile(fullPath, 'utf-8');

  // Attempt to apply the unified diff
  const patched = Diff.applyPatch(originalContent, diffText);

  if (patched === false) {
    return {
      success: false,
      path: relPath,
      error: 'Failed to apply diff: hunks did not match target file content. Verify line offsets and surrounding context.'
    };
  }

  // Write patched content to file
  await fs.writeFile(fullPath, patched, 'utf-8');

  return {
    success: true,
    path: relPath,
    message: 'Unified diff successfully applied.'
  };
}

/**
 * Format file using clang-format or fallback
 */
export async function formatFile(relPath) {
  const fullPath = vfs.resolvePath(relPath);
  if (!fsSync.existsSync(fullPath)) {
    throw new Error(`File "${relPath}" not found.`);
  }

  // Check if clang-format is installed
  try {
    await execPromise(`which clang-format`);
    const { stdout } = await execPromise(`clang-format "${fullPath}"`);
    await fs.writeFile(fullPath, stdout, 'utf-8');
    return {
      success: true,
      path: relPath,
      formatter: 'clang-format',
      message: 'Formatted successfully using clang-format.'
    };
  } catch {
    // Basic formatting fallback: normalize trailing spaces and final newline
    const content = await fs.readFile(fullPath, 'utf-8');
    const cleaned = content
      .split('\n')
      .map(line => line.trimEnd())
      .join('\n')
      .trimEnd() + '\n';

    await fs.writeFile(fullPath, cleaned, 'utf-8');
    return {
      success: true,
      path: relPath,
      formatter: 'standard-whitespace',
      message: 'Normalized trailing whitespace and newlines.'
    };
  }
}

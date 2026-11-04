import { exec } from 'child_process';
import util from 'util';
import fsSync from 'fs';
import path from 'path';
import { vfs, WORKSPACE_ROOT } from './vfs.js';

const execPromise = util.promisify(exec);

/**
 * Lint / Syntax Check for C/C++ and JS files
 */
export async function lintFileOrProject(relPath = '.') {
  const fullPath = vfs.resolvePath(relPath);
  if (!fsSync.existsSync(fullPath)) {
    throw new Error(`Path "${relPath}" not found.`);
  }

  const stat = fsSync.statSync(fullPath);
  const errors = [];
  const warnings = [];

  if (stat.isFile()) {
    const ext = path.extname(relPath).toLowerCase();

    if (['.c', '.cpp', '.h', '.hpp'].includes(ext)) {
      // Clang syntax-only check
      const dir = path.dirname(fullPath);
      const incFlags = `-I"${dir}" -I"${path.join(dir, 'include')}" -I"${path.join(dir, '..', 'include')}"`;
      const cmd = `clang -fsyntax-only ${incFlags} "${fullPath}" 2>&1`;

      try {
        const { stdout, stderr } = await execPromise(cmd, { cwd: WORKSPACE_ROOT });
        const output = (stdout || stderr || '').trim();
        parseClangDiagnostics(output, errors, warnings);
      } catch (err) {
        const output = (err.stdout || err.stderr || err.message || '').trim();
        parseClangDiagnostics(output, errors, warnings);
      }
    } else if (['.js', '.mjs'].includes(ext)) {
      // Node syntax check
      try {
        await execPromise(`node -c "${fullPath}"`);
      } catch (err) {
        errors.push({
          file: relPath,
          line: 1,
          severity: 'error',
          message: err.message
        });
      }
    }
  }

  return {
    success: errors.length === 0,
    target: relPath,
    errorCount: errors.length,
    warningCount: warnings.length,
    errors,
    warnings
  };
}

function parseClangDiagnostics(rawOutput, errors, warnings) {
  const lines = rawOutput.split('\n');
  const diagRegex = /^(.*?):(\d+):(\d+):\s+(error|warning|fatal error):\s+(.*)$/;

  for (const line of lines) {
    const match = line.match(diagRegex);
    if (match) {
      const item = {
        file: path.basename(match[1]),
        line: Number(match[2]),
        column: Number(match[3]),
        severity: match[4] === 'warning' ? 'warning' : 'error',
        message: match[5].trim()
      };
      if (item.severity === 'warning') {
        warnings.push(item);
      } else {
        errors.push(item);
      }
    }
  }
}

/**
 * Trigger Project Build (e.g., make, npm run build)
 */
export async function buildProject(relDir = '.', command = 'make') {
  const fullDir = vfs.resolvePath(relDir);
  if (!fsSync.existsSync(fullDir)) {
    throw new Error(`Directory "${relDir}" not found.`);
  }

  const startTime = Date.now();
  console.log(`🔨 [Headless IDE] Executing build in "${relDir}": ${command}`);

  try {
    const { stdout, stderr } = await execPromise(command, {
      cwd: fullDir,
      timeout: 120000,
      maxBuffer: 1024 * 1024
    });

    const durationMs = Date.now() - startTime;
    return {
      success: true,
      directory: relDir,
      command,
      exitCode: 0,
      durationMs,
      stdout: (stdout || '').trim(),
      stderr: (stderr || '').trim()
    };
  } catch (err) {
    const durationMs = Date.now() - startTime;
    return {
      success: false,
      directory: relDir,
      command,
      exitCode: err.code || 1,
      durationMs,
      stdout: (err.stdout || '').trim(),
      stderr: (err.stderr || err.message || '').trim()
    };
  }
}

/**
 * Trigger Project Test
 */
export async function testProject(relDir = '.', command = 'npm test') {
  const fullDir = vfs.resolvePath(relDir);
  if (!fsSync.existsSync(fullDir)) {
    throw new Error(`Directory "${relDir}" not found.`);
  }

  const startTime = Date.now();
  try {
    const { stdout, stderr } = await execPromise(command, {
      cwd: fullDir,
      timeout: 60000,
      maxBuffer: 512 * 1024
    });

    return {
      success: true,
      directory: relDir,
      command,
      durationMs: Date.now() - startTime,
      output: (stdout || stderr || '').trim()
    };
  } catch (err) {
    return {
      success: false,
      directory: relDir,
      command,
      exitCode: err.code || 1,
      durationMs: Date.now() - startTime,
      error: (err.stderr || err.stdout || err.message || '').trim()
    };
  }
}

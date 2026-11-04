import { vfs } from './vfs.js';
import { getFileSymbols, findSymbolReferences } from './symbolParser.js';
import { replaceAllInFile, applyDiffToFile } from './manipulator.js';
import { lintFileOrProject, buildProject } from './feedback.js';
import * as Diff from 'diff';
import fs from 'fs/promises';
import fsSync from 'fs';

async function runTestSuite() {
  console.log('🧪 ========================================================');
  console.log('🧪 Starting Headless IDE Comprehensive Verification Suite');
  console.log('🧪 ========================================================\n');

  let passed = 0;
  let total = 0;

  const assert = (condition, title) => {
    total++;
    if (condition) {
      console.log(`✅ [PASS] ${title}`);
      passed++;
    } else {
      console.error(`❌ [FAIL] ${title}`);
    }
  };

  try {
    const targetDir = process.env.TEST_PROJECT_DIR || '.';
    const targetSample = process.env.TEST_SAMPLE_FILE || 'headless_ide/vfs.js';
    const targetSymbol = process.env.TEST_SYMBOL || 'VirtualFileSystem';

    // 1. VFS Project Structure
    console.log('📁 Test 1: VFS Project Structure');
    const tree = await vfs.getProjectStructure(targetDir, 3);
    assert(tree && tree.type === 'directory', 'VFS returns structured tree');
    assert(tree.children.length > 0, 'VFS identifies directory children');

    // 2. Symbol Parsing (C/C++ or JS)
    console.log('\n🔍 Test 2: Semantic Symbol Extraction');
    const symbolResult = await getFileSymbols(targetSample);
    assert(symbolResult.totalSymbols > 0, `Extracted ${symbolResult.totalSymbols} symbols from ${targetSample}`);
    const hasInit = symbolResult.symbols.some(s => s.name === targetSymbol);
    assert(hasInit, `Found ${targetSymbol} symbol definition`);

    // 3. Symbol Reference Tracing
    console.log('\n🔗 Test 3: Cross-Project Reference Tracing');
    const refResult = await findSymbolReferences(targetSymbol, targetDir);
    assert(refResult.referenceCount >= 1, `Found ${refResult.referenceCount} references to ${targetSymbol} across project`);

    // 4. File Context Window
    console.log('\n🪟 Test 4: File Context Window');
    const ctx = await vfs.getFileContext(targetSample, 15, 2);
    assert(ctx.lines.length === 5, 'Context window returns correct line slice');
    assert(ctx.lines.some(l => l.isTarget && l.line === 15), 'Target line is accurately flagged');

    // 5. Atomic Manipulation & Unified Diff
    console.log('\n✏️ Test 5: Atomic Replacement & Unified Diff Application');
    const tempFile = 'headless_ide/temp_test.txt';
    await fs.writeFile(tempFile, 'Line 1\nAlpha\nLine 3\n');

    const replaceRes = await replaceAllInFile(tempFile, 'Alpha', 'Beta');
    assert(replaceRes.count === 1, 'Regex replace matches exactly 1 occurrence');
    assert(replaceRes.diff.includes('+Beta'), 'Generated unified diff contains replacement hunk');

    // Revert using unified diff
    const reverseDiff = Diff.createTwoFilesPatch('a', 'b', 'Line 1\nBeta\nLine 3\n', 'Line 1\nAlpha\nLine 3\n');
    const patchRes = await applyDiffToFile(tempFile, reverseDiff);
    assert(patchRes.success, 'Unified diff applied cleanly');

    const revertedContent = await fs.readFile(tempFile, 'utf-8');
    assert(revertedContent.includes('Alpha'), 'File content restored accurately by diff');
    await fs.unlink(tempFile);

    // 6. Linting Diagnostics
    console.log('\n🔬 Test 6: Syntax & Linting Engine');
    const lintRes = await lintFileOrProject(targetSample);
    assert(lintRes.target.includes(targetSample), 'Linter processes source file');
    assert(Array.isArray(lintRes.errors), 'Linter returns structured errors array');

    // 7. Project Build Execution
    console.log('\n🔨 Test 7: Project Build Trigger');
    const buildCmd = process.env.TEST_BUILD_CMD || 'node -v';
    const buildRes = await buildProject(targetDir, buildCmd);
    assert(buildRes.durationMs >= 0, `Build executed and measured (${buildRes.durationMs}ms)`);
    assert(typeof buildRes.exitCode === 'number', 'Build returns process exit code');

    console.log(`\n========================================================`);
    console.log(`🎉 Suite Completed: ${passed}/${total} Tests Passed`);
    console.log(`========================================================\n`);

    if (passed !== total) {
      process.exit(1);
    }
  } catch (err) {
    console.error('Fatal test error:', err);
    process.exit(1);
  }
}

runTestSuite();

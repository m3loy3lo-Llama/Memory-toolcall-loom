/**
 * Distillation Engine & Tiered Memory System
 * 
 * Implements:
 * 1. Reasoning-Centric Compaction: Captures the "why" (heuristics, decision paths, and engineering intent)
 * 2. Tiered Memory Distillation: Hot (Active context) -> Warm (Reasoning Compactions) -> Cold (LoRA weight updates)
 * 3. Action Chain Training: Captures and learns transition connective tissue between tool calls
 * 4. Contextual Branching: Swappable project-specific weight adapters with base heuristic preservation
 */

import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { EMBEDDING_DIM, generateEmbedding, cosineSimilarity } from './memoryManager.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Rank of low-rank projection adapter (d=1280, r=16)
export const ADAPTER_RANK = 16;
export const ADAPTER_ALPHA = 1.0;

/**
 * ---------------------------------------------------------------------------
 * 1. ColdWeightAdapter: Low-Rank Matrix Adapter (LoRA for 1280-dim Latents)
 * ---------------------------------------------------------------------------
 * Projects embeddings through W_A (1280 -> 16) and W_B (16 -> 1280).
 * Supports online gradient training, weight serialization, and contextual branching.
 */
export class ColdWeightAdapter {
  constructor(branchName = 'general_heuristics', storageDir = path.resolve(__dirname, 'trajectories/branches')) {
    this.branchName = branchName;
    this.storageDir = storageDir;
    this.rank = ADAPTER_RANK;
    this.dim = EMBEDDING_DIM;
    this.alpha = ADAPTER_ALPHA;
    this.learningRate = 0.01;
    this.trainSteps = 0;
    this.heuristicsLearned = [];

    // Initialize low-rank matrices
    // W_A: Gaussian initialization with std = 1/sqrt(dim)
    // W_B: Zero initialization (LoRA standard: starts as identity / zero delta)
    this.W_A = new Float32Array(this.dim * this.rank);
    this.W_B = new Float32Array(this.rank * this.dim);
    this._initMatrices();

    this._ensureDir();
    this.loadWeightsSync();
  }

  _ensureDir() {
    try {
      if (!fsSync.existsSync(this.storageDir)) {
        fsSync.mkdirSync(this.storageDir, { recursive: true });
      }
    } catch {
      // Non-fatal
    }
  }

  _initMatrices() {
    const std = 1.0 / Math.sqrt(this.dim);
    for (let i = 0; i < this.W_A.length; i++) {
      // Box-Muller normal distribution
      const u1 = Math.max(1e-7, Math.random());
      const u2 = Math.random();
      this.W_A[i] = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * std;
    }
    this.W_B.fill(0); // Zero initialization for W_B
  }

  getFilePath() {
    return path.join(this.storageDir, `adapter_${this.branchName}.json`);
  }

  /**
   * Forward pass: projects input vector x (1280) through adapter
   * output = x + alpha * (x * W_A) * W_B
   */
  adapt(x) {
    if (!x || x.length !== this.dim) return x;

    // 1. h = x * W_A (1 x rank)
    const h = new Float32Array(this.rank);
    for (let r = 0; r < this.rank; r++) {
      let sum = 0;
      for (let d = 0; d < this.dim; d++) {
        sum += x[d] * this.W_A[d * this.rank + r];
      }
      h[r] = sum;
    }

    // 2. delta = alpha * (h * W_B) (1 x dim)
    const out = new Float32Array(this.dim);
    for (let d = 0; d < this.dim; d++) {
      let sum = 0;
      for (let r = 0; r < this.rank; r++) {
        sum += h[r] * this.W_B[r * this.dim + d];
      }
      out[d] = x[d] + this.alpha * sum;
    }

    // L2 Normalize
    let norm = 0;
    for (let d = 0; d < this.dim; d++) norm += out[d] * out[d];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let d = 0; d < this.dim; d++) out[d] /= norm;
    }

    return out;
  }

  /**
   * Online training step: Backpropagate reconstruction/transition loss into W_A and W_B
   * min || adapt(x) - target ||^2
   */
  trainPair(inputVec, targetVec, learningRate = this.learningRate) {
    if (!inputVec || !targetVec) return 0;

    // 1. Forward
    const h = new Float32Array(this.rank);
    for (let r = 0; r < this.rank; r++) {
      let sum = 0;
      for (let d = 0; d < this.dim; d++) {
        sum += inputVec[d] * this.W_A[d * this.rank + r];
      }
      h[r] = sum;
    }

    const delta = new Float32Array(this.dim);
    for (let d = 0; d < this.dim; d++) {
      let sum = 0;
      for (let r = 0; r < this.rank; r++) {
        sum += h[r] * this.W_B[r * this.dim + d];
      }
      delta[d] = this.alpha * sum;
    }

    // Loss grad wrt delta: error = (input + delta) - target
    let loss = 0;
    const gradDelta = new Float32Array(this.dim);
    for (let d = 0; d < this.dim; d++) {
      const err = (inputVec[d] + delta[d]) - targetVec[d];
      loss += err * err;
      gradDelta[d] = 2.0 * err * this.alpha;
    }

    // Gradient wrt W_B and h
    const gradH = new Float32Array(this.rank);
    for (let r = 0; r < this.rank; r++) {
      let sumH = 0;
      for (let d = 0; d < this.dim; d++) {
        const g = gradDelta[d];
        sumH += g * this.W_B[r * this.dim + d];
        // Update W_B: W_B[r, d] -= lr * h[r] * gradDelta[d]
        this.W_B[r * this.dim + d] -= learningRate * h[r] * g;
      }
      gradH[r] = sumH;
    }

    // Update W_A: W_A[d, r] -= lr * inputVec[d] * gradH[r]
    for (let d = 0; d < this.dim; d++) {
      for (let r = 0; r < this.rank; r++) {
        this.W_A[d * this.rank + r] -= learningRate * inputVec[d] * gradH[r];
      }
    }

    this.trainSteps++;
    return loss / this.dim;
  }

  /**
   * Distill a collection of vectors (observations, mutations, action transitions) into weights
   */
  distillBatch(pairs, epochs = 2) {
    if (!pairs || pairs.length === 0) return { steps: 0, avgLoss: 0 };
    let totalLoss = 0;
    let count = 0;

    for (let ep = 0; ep < epochs; ep++) {
      for (const pair of pairs) {
        const inputVec = pair.input || pair.embedding;
        const targetVec = pair.target || pair.embedding;
        if (inputVec && targetVec) {
          const l = this.trainPair(inputVec, targetVec);
          totalLoss += l;
          count++;
        }
      }
    }

    this.saveWeightsSync();
    return {
      steps: count,
      avgLoss: count > 0 ? (totalLoss / count) : 0,
      totalTrainSteps: this.trainSteps
    };
  }

  saveWeightsSync() {
    try {
      const data = {
        branchName: this.branchName,
        dim: this.dim,
        rank: this.rank,
        trainSteps: this.trainSteps,
        heuristicsLearned: this.heuristicsLearned,
        updatedAt: new Date().toISOString(),
        // Encode weights as base64 for fast compact JSON storage
        W_A: Buffer.from(this.W_A.buffer).toString('base64'),
        W_B: Buffer.from(this.W_B.buffer).toString('base64')
      };
      fsSync.writeFileSync(this.getFilePath(), JSON.stringify(data), 'utf-8');
    } catch (err) {
      console.warn(`⚠️ Failed to save adapter weights for branch "${this.branchName}":`, err.message);
    }
  }

  loadWeightsSync() {
    const file = this.getFilePath();
    if (!fsSync.existsSync(file)) return false;

    try {
      const parsed = JSON.parse(fsSync.readFileSync(file, 'utf-8'));
      if (parsed.W_A && parsed.W_B) {
        const bufA = Buffer.from(parsed.W_A, 'base64');
        const bufB = Buffer.from(parsed.W_B, 'base64');
        this.W_A = new Float32Array(bufA.buffer, bufA.byteOffset, bufA.length / Float32Array.BYTES_PER_ELEMENT);
        this.W_B = new Float32Array(bufB.buffer, bufB.byteOffset, bufB.length / Float32Array.BYTES_PER_ELEMENT);
        this.trainSteps = parsed.trainSteps || 0;
        this.heuristicsLearned = parsed.heuristicsLearned || [];
        return true;
      }
    } catch (err) {
      console.warn(`⚠️ Failed to load adapter weights from ${file}:`, err.message);
    }
    return false;
  }
}

/**
 * ---------------------------------------------------------------------------
 * 2. ActionChainTracker: Connective Tissue & Workflow Learning
 * ---------------------------------------------------------------------------
 * Tracks multi-step tool call sequences, calculates transition heuristics,
 * and trains the sub-agent on logical workflow chains.
 */
export class ActionChainTracker {
  constructor() {
    this.chain = []; // Current turn chain: [{ tool, args, summary, resultStatus, embedding, timestamp }]
    this.completedChains = []; // History of past action sequences
  }

  recordAction(toolName, args, resultSummary, isSuccess = true) {
    const actionText = `Action: ${toolName} with ${JSON.stringify(args || {}).slice(0, 150)} -> ${resultSummary}`;
    const embedding = generateEmbedding(actionText);

    const step = {
      tool: toolName,
      args: typeof args === 'object' ? args : { raw: args },
      summary: resultSummary,
      isSuccess,
      embedding: Array.from(embedding),
      timestamp: Date.now()
    };

    this.chain.push(step);
    return step;
  }

  /**
   * Generates training pairs representing transition dynamics (tool_{t-1} -> tool_t)
   */
  extractTransitionPairs() {
    const pairs = [];
    if (this.chain.length < 2) return pairs;

    for (let i = 0; i < this.chain.length - 1; i++) {
      const prev = this.chain[i];
      const next = this.chain[i + 1];

      pairs.push({
        input: prev.embedding,
        target: next.embedding,
        sentence: `Transition: ${prev.tool} -> ${next.tool} [${next.summary.slice(0, 100)}]`,
        metadata: {
          fromTool: prev.tool,
          toTool: next.tool,
          transitionType: 'action_chain'
        }
      });
    }

    return pairs;
  }

  /**
   * Suggests next logical proactive tool based on workflow motifs
   */
  suggestNextAction() {
    if (this.chain.length === 0) {
      return { tool: 'list_dir', reason: 'Initial workspace orientation and directory discovery.' };
    }

    const last = this.chain[this.chain.length - 1];

    switch (last.tool) {
      case 'list_dir':
        return { tool: 'grep_search', reason: 'Locate key project symbols or implementation keywords in discovered files.' };
      case 'grep_search':
        return { tool: 'view_file', reason: 'Read the lines containing the matched symbol with full context.' };
      case 'ide_get_symbols':
        return { tool: 'ide_find_references', reason: 'Trace symbol usage and call-sites across the project before modifying.' };
      case 'view_file':
      case 'ide_get_context':
        return { tool: 'ide_apply_diff', reason: 'Apply surgical, atomic diff modifications to the inspected file.' };
      case 'ide_apply_diff':
      case 'replace_file_content':
      case 'write_file':
        return { tool: 'ide_lint', reason: 'Verify syntax and check for compiler diagnostics on modified files.' };
      case 'ide_lint':
        return { tool: 'ide_build', reason: 'Execute project build or tests to ensure overall system integrity.' };
      default:
        return { tool: 'retrieve_memory', reason: 'Inspect live project memory or state changes.' };
    }
  }

  finalizeChain(outcomeSummary) {
    if (this.chain.length > 0) {
      this.completedChains.push({
        steps: [...this.chain],
        outcome: outcomeSummary,
        length: this.chain.length,
        timestamp: Date.now()
      });
      if (this.completedChains.length > 20) this.completedChains.shift();
      const current = [...this.chain];
      this.chain = [];
      return current;
    }
    return [];
  }

  getRecentChainsMarkdown() {
    if (this.completedChains.length === 0 && this.chain.length === 0) return '';

    let md = `\n#### 🔗 Learned Action Chains & Workflow Connective Tissue:\n`;
    const chainsToShow = this.completedChains.slice(-3);

    chainsToShow.forEach((c, idx) => {
      const flow = c.steps.map(s => `\`${s.tool}\``).join(' ➔ ');
      md += `- **Pattern ${idx + 1}**: ${flow} (${c.outcome || 'Success'})\n`;
    });

    if (this.chain.length > 0) {
      const activeFlow = this.chain.map(s => `\`${s.tool}\``).join(' ➔ ');
      md += `- **Active Chain**: ${activeFlow} *(Next suggested: \`${this.suggestNextAction().tool}\`)*\n`;
    }

    return md;
  }
}

/**
 * ---------------------------------------------------------------------------
 * 3. CompactionManager: Reasoning-Centric Summarization & Tiered Distillation
 * ---------------------------------------------------------------------------
 */
export class CompactionManager {
  constructor(options = {}) {
    this.storageDir = options.storageDir || path.resolve(__dirname, 'trajectories');
    this.activeBranch = options.defaultBranch || 'general_heuristics';
    
    // Core adapters: Base heuristics + Project-specific branch
    this.baseAdapter = new ColdWeightAdapter('general_heuristics', path.join(this.storageDir, 'branches'));
    this.projectAdapter = new ColdWeightAdapter(this.activeBranch, path.join(this.storageDir, 'branches'));
    
    this.actionChainTracker = new ActionChainTracker();
    this.compactions = []; // Warm memory compaction records
    this.maxWarmRecords = 15;

    this._ensureStorage();
    this._loadCompactions();
  }

  _ensureStorage() {
    try {
      const compDir = path.join(this.storageDir, 'compactions');
      if (!fsSync.existsSync(compDir)) {
        fsSync.mkdirSync(compDir, { recursive: true });
      }
    } catch {
      // Non-fatal
    }
  }

  async _loadCompactions() {
    try {
      const file = path.join(this.storageDir, 'compactions', `compactions_${this.activeBranch}.json`);
      if (fsSync.existsSync(file)) {
        const data = JSON.parse(await fs.readFile(file, 'utf-8'));
        this.compactions = data || [];
      }
    } catch {
      this.compactions = [];
    }
  }

  async _saveCompactions() {
    try {
      const file = path.join(this.storageDir, 'compactions', `compactions_${this.activeBranch}.json`);
      await fs.writeFile(file, JSON.stringify(this.compactions, null, 2), 'utf-8');
    } catch (err) {
      console.warn('⚠️ Failed to save compactions log:', err.message);
    }
  }

  /**
   * Switch or create contextual memory branch for a specific task or project
   */
  async switchBranch(newBranchName) {
    if (!newBranchName || newBranchName === this.activeBranch) {
      return { success: true, branch: this.activeBranch, message: 'Already on active branch' };
    }

    console.log(`🌿 [Compaction] Switching contextual memory branch from "${this.activeBranch}" to "${newBranchName}"`);
    
    // Save current branch
    this.projectAdapter.saveWeightsSync();
    await this._saveCompactions();

    this.activeBranch = newBranchName;
    this.projectAdapter = new ColdWeightAdapter(this.activeBranch, path.join(this.storageDir, 'branches'));
    await this._loadCompactions();

    return {
      success: true,
      branch: this.activeBranch,
      trainSteps: this.projectAdapter.trainSteps,
      heuristicsCount: this.projectAdapter.heuristicsLearned.length
    };
  }

  /**
   * List all available contextual memory branches
   */
  async listBranches() {
    const branchesDir = path.join(this.storageDir, 'branches');
    try {
      if (!fsSync.existsSync(branchesDir)) return [this.activeBranch];
      const files = await fs.readdir(branchesDir);
      const branches = files
        .filter(f => f.startsWith('adapter_') && f.endsWith('.json'))
        .map(f => f.replace(/^adapter_/, '').replace(/\.json$/, ''));
      
      if (!branches.includes('general_heuristics')) branches.unshift('general_heuristics');
      if (!branches.includes(this.activeBranch)) branches.push(this.activeBranch);
      return [...new Set(branches)];
    } catch {
      return [this.activeBranch];
    }
  }

  /**
   * SMART COMPACTION:
   * 1. Summarizes the "why": heuristics, decisions, intent, and workspace state
   * 2. Distills warm session into cold LoRA weight updates
   * 3. Trains sub-agent on action chains and connective tissue
   * 4. Produces a lightweight, crystal-clear handover context for the incoming agent
   */
  async compactContext({
    reason = 'Context milestone reached',
    milestone = 'Task completed',
    heuristics = [],
    decisionPath = '',
    memoryManager = null
  } = {}) {
    const timestamp = new Date().toISOString();
    const id = `compact_${Date.now()}`;

    console.log(`🧹 [Smart Compaction] Initiating compaction (${reason}) on branch "${this.activeBranch}"...`);

    // 1. Gather recent observations & mutations
    const recentMutations = memoryManager?.mutations || [];
    const recentObservations = memoryManager?.observations || [];
    
    // 2. Extract action chain transitions
    const transitionPairs = this.actionChainTracker.extractTransitionPairs();
    const completedChain = this.actionChainTracker.finalizeChain(milestone);

    // 3. Compile Reasoning-Centric Summary
    const structuredSummary = {
      id,
      timestamp,
      branch: this.activeBranch,
      reason,
      milestone,
      decisionPath: decisionPath || 'Evaluated surgical tool operations against workspace safety rules.',
      heuristics: Array.isArray(heuristics) && heuristics.length > 0 
        ? heuristics 
        : [
            'Use surgical diffs (ide_apply_diff) over whole file overwrites for precision.',
            'Run ide_lint and ide_build immediately following atomic file mutations.',
            'Query symbols with ide_get_symbols prior to refactoring to preserve signatures.'
          ],
      filesModified: [...new Set(recentMutations.map(m => m.file).filter(Boolean))],
      actionChainLength: completedChain.length,
      toolsUsed: [...new Set(completedChain.map(s => s.tool))]
    };

    // Store into warm memory records
    this.compactions.unshift(structuredSummary);
    if (this.compactions.length > this.maxWarmRecords) this.compactions.pop();
    await this._saveCompactions();

    // 4. COLD TIER DISTILLATION:
    // Generate embeddings for the compaction summary & heuristics
    const summaryText = `Compaction ${milestone}: ${structuredSummary.decisionPath}. Heuristics: ${structuredSummary.heuristics.join('; ')}`;
    const summaryEmbedding = generateEmbedding(summaryText);

    const distillationPairs = [...transitionPairs];
    
    // Add summary embedding as a high-value anchor
    distillationPairs.push({
      input: summaryEmbedding,
      target: summaryEmbedding,
      sentence: summaryText,
      metadata: { type: 'milestone_summary', milestone }
    });

    // Add heuristics pairs
    structuredSummary.heuristics.forEach(h => {
      const hVec = generateEmbedding(`Heuristic Rule: ${h}`);
      distillationPairs.push({
        input: hVec,
        target: hVec,
        sentence: h,
        metadata: { type: 'heuristic' }
      });
      if (!this.projectAdapter.heuristicsLearned.includes(h)) {
        this.projectAdapter.heuristicsLearned.push(h);
      }
    });

    // Train the Cold Weight Adapter on this batch (Online LoRA step)
    const trainResult = this.projectAdapter.distillBatch(distillationPairs, 3);
    console.log(`🧊 [Cold Distillation] Trained ${trainResult.steps} steps into adapter "${this.activeBranch}" (loss: ${trainResult.avgLoss.toFixed(5)})`);

    // 5. Build Helper Prompt / Handover Briefing for the Next Agent / Sub-Agent
    const handoverBrief = this.generateHandoverMarkdown(structuredSummary);

    const checkpoint = {
      id,
      timestamp,
      branch: this.activeBranch,
      milestone: structuredSummary.milestone,
      reason: structuredSummary.reason,
      decisionPath: structuredSummary.decisionPath,
      heuristics: structuredSummary.heuristics,
      filesModified: structuredSummary.filesModified,
      actionChainLength: structuredSummary.actionChainLength,
      distillationSteps: trainResult.steps,
      handoverBrief
    };

    return {
      success: true,
      compactionId: id,
      summary: structuredSummary,
      distillation: trainResult,
      handoverBrief,
      checkpoint
    };
  }

  /**
   * Generates a lightweight briefing for the Senior Developer / Next Agent
   */
  generateHandoverMarkdown(latestCompaction) {
    const c = latestCompaction || this.compactions[0];
    if (!c) return '';

    let md = `\n### 🧭 Workspace Orientation & Distilled Logic (Handover from Lead Architect):\n`;
    md += `- **Active Memory Branch**: \`${c.branch}\`\n`;
    md += `- **Last Milestone**: ${c.milestone}\n`;
    md += `- **Architectural Intent & Decisions**: ${c.decisionPath}\n`;
    
    if (c.heuristics && c.heuristics.length > 0) {
      md += `- **Learned Engineering Heuristics**:\n`;
      c.heuristics.forEach(h => {
        md += `  * 💡 ${h}\n`;
      });
    }

    if (c.filesModified && c.filesModified.length > 0) {
      md += `- **Active Files Under Modification**: ${c.filesModified.map(f => `\`${f}\``).join(', ')}\n`;
    }

    const nextSug = this.actionChainTracker.suggestNextAction();
    md += `- **Recommended Next Action**: Call \`${nextSug.tool}\` (${nextSug.reason})\n`;

    return md;
  }

  getTiersStatus() {
    return {
      activeBranch: this.activeBranch,
      warmCompactionsCount: this.compactions.length,
      coldTrainSteps: this.projectAdapter.trainSteps,
      heuristicsCount: this.projectAdapter.heuristicsLearned.length,
      recentHeuristics: this.projectAdapter.heuristicsLearned.slice(-5),
      activeChainLength: this.actionChainTracker.chain.length
    };
  }
}

export const compactionManager = new CompactionManager();

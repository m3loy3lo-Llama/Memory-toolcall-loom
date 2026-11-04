import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { modelManager } from './modelManager.js';
import { runAgentLoop } from './agent.js';
import { TOOL_DEFINITIONS, WORKSPACE_DIR, listAllFiles } from './tools.js';
import { memoryManager } from './memoryManager.js';
import { compactionManager } from './distillationEngine.js';
import { runSubAgent } from './subAgent.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ---------------------------------------------------------------------------
// API Endpoints
// ---------------------------------------------------------------------------

/**
 * Health & Model Status
 */
app.get('/api/status', (req, res) => {
  const status = modelManager.getStatus();
  res.json({
    status: status.isReady ? 'ready' : (status.isStarting ? 'starting' : 'stopped'),
    model: status.model,
    hasVision: status.hasVision,
    workspaceDir: WORKSPACE_DIR,
    tools: TOOL_DEFINITIONS.map(t => t.function.name),
    ctxSize: status.ctxSize,
    workingMemory: {
      observations: memoryManager?.observations?.length || 0,
      mutations: memoryManager?.mutations?.length || 0,
      trajectories: (memoryManager?.trajectories?.length || memoryManager?.history?.length || 0)
    }
  });
});

/**
 * Tool Definitions
 */
app.get('/api/tools', (req, res) => {
  res.json({ tools: TOOL_DEFINITIONS });
});

/**
 * List Discoverable Models & Active Model
 */
app.get('/api/models', async (req, res) => {
  try {
    const models = await modelManager.listAvailableModels();
    res.json({
      success: true,
      activeModel: modelManager.getStatus().model,
      activeModelPath: modelManager.getStatus().modelPath,
      models
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

/**
 * Hot-Swap Model at Runtime
 */
app.post('/api/models/switch', async (req, res) => {
  try {
    const { modelPath, mmprojPath, ctxSize, gpuLayers } = req.body;
    if (!modelPath) {
      return res.status(400).json({ success: false, error: 'modelPath is required' });
    }
    console.log(`🔄 Request to hot-swap model to: ${modelPath}`);
    const status = await modelManager.switchModel({ modelPath, mmprojPath, ctxSize, gpuLayers });
    res.json({ success: true, status });
  } catch (err) {
    console.error('❌ Failed to hot-swap model:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

/**
 * Chat & Agent Execution
 */
app.post('/api/chat', async (req, res) => {
  try {
    const { messages, temperature = 0.2, maxSteps = 30 } = req.body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Messages array is required.' });
    }

    console.log(`💬 [/api/chat] Incoming request with ${messages.length} messages (maxSteps: ${maxSteps})`);

    const assistantMessage = await runAgentLoop({
      messages,
      temperature,
      maxSteps: Number(maxSteps) || 30
    });

    res.json({
      success: true,
      message: assistantMessage
    });
  } catch (err) {
    console.error('❌ [/api/chat] Error executing agent loop:', err);
    res.status(500).json({
      success: false,
      error: err.message || 'Internal server error while processing agent request.'
    });
  }
});

/**
 * Clear / Reset Session
 */
app.post('/api/clear', (req, res) => {
  console.log('🧹 [/api/clear] Session reset requested.');
  if (memoryManager && typeof memoryManager.clear === 'function') {
    memoryManager.clear();
  }
  res.json({ success: true, message: 'Session and working memory reset successfully.' });
});

/**
 * Inspect Active Working Memory & Trainable Embeddings
 */
app.get('/api/memory', (req, res) => {
  res.json({
    mutations: memoryManager?.mutations || [],
    observations: memoryManager?.observations || [],
    trajectoriesCount: (memoryManager?.trajectories?.length || memoryManager?.history?.length || 0),
    journal: memoryManager?.getJournalMarkdown ? memoryManager.getJournalMarkdown() : (memoryManager?.getMemorySummary ? memoryManager.getMemorySummary() : '')
  });
});

/**
 * Tiered Memory Status (Hot, Warm, Cold)
 */
app.get('/api/memory/tiers', (req, res) => {
  const tiers = compactionManager.getTiersStatus();
  res.json({
    success: true,
    tiers: {
      hot: {
        liveMutations: memoryManager?.mutations?.length || 0,
        liveObservations: memoryManager?.observations?.length || 0,
        liveTrajectories: memoryManager?.trajectories?.length || 0
      },
      warm: {
        activeBranch: tiers.activeBranch,
        compactionsCount: tiers.warmCompactionsCount,
        latestCompaction: compactionManager.compactions[0] || null
      },
      cold: {
        branch: tiers.activeBranch,
        trainSteps: tiers.coldTrainSteps,
        heuristicsCount: tiers.heuristicsCount,
        recentHeuristics: tiers.recentHeuristics
      },
      actionChains: {
        activeChainLength: tiers.activeChainLength,
        suggestedNext: compactionManager.actionChainTracker.suggestNextAction()
      },
      checkpoint: compactionManager.compactions[0] ? {
        id: compactionManager.compactions[0].id,
        timestamp: compactionManager.compactions[0].timestamp,
        branch: compactionManager.compactions[0].branch,
        milestone: compactionManager.compactions[0].milestone,
        decisionPath: compactionManager.compactions[0].decisionPath,
        heuristics: compactionManager.compactions[0].heuristics
      } : null
    }
  });
});

/**
 * Get Latest Context Checkpoint
 */
app.get('/api/memory/checkpoint', (req, res) => {
  const latestComp = compactionManager.compactions[0] || null;
  if (!latestComp) {
    return res.json({ success: true, checkpoint: null });
  }
  res.json({
    success: true,
    checkpoint: {
      id: latestComp.id,
      timestamp: latestComp.timestamp,
      branch: latestComp.branch,
      milestone: latestComp.milestone,
      decisionPath: latestComp.decisionPath,
      heuristics: latestComp.heuristics,
      filesModified: latestComp.filesModified,
      handoverBrief: compactionManager.generateHandoverMarkdown(latestComp)
    }
  });
});

/**
 * Trigger Smart Compaction Manually
 */
app.post('/api/memory/compact', async (req, res) => {
  try {
    const { reason, milestone, decisionPath, heuristics } = req.body;
    console.log(`🧹 [/api/memory/compact] Compaction requested: ${milestone || reason}`);
    const result = await compactionManager.compactContext({
      reason,
      milestone,
      decisionPath,
      heuristics,
      memoryManager
    });
    res.json(result);
  } catch (err) {
    console.error('❌ Compaction failed:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

/**
 * List Contextual Memory Branches
 */
app.get('/api/memory/branches', async (req, res) => {
  try {
    const branches = await compactionManager.listBranches();
    res.json({
      success: true,
      activeBranch: compactionManager.activeBranch,
      branches
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

/**
 * Switch Contextual Memory Branch
 */
app.post('/api/memory/branch', async (req, res) => {
  try {
    const { branch_name } = req.body;
    if (!branch_name) {
      return res.status(400).json({ error: 'branch_name is required' });
    }
    const result = await compactionManager.switchBranch(branch_name);
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

/**
 * Direct Sub-Agent Delegation Endpoint
 */
app.post('/api/agent/delegate', async (req, res) => {
  try {
    const { task, instructions, targetFiles, maxSteps } = req.body;
    if (!task) {
      return res.status(400).json({ error: 'task is required' });
    }
    console.log(`👷 [/api/agent/delegate] Delegating sub-agent task: "${task}"`);
    const result = await runSubAgent({
      task,
      instructions,
      targetFiles,
      maxSteps: Number(maxSteps) || 10
    });
    res.json({ success: true, result });
  } catch (err) {
    console.error('❌ Sub-agent delegation error:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ---------------------------------------------------------------------------
// Server Startup
// ---------------------------------------------------------------------------

app.listen(PORT, async () => {
  console.log(`\n======================================================`);
  console.log(`🚀 Local AI Agent Server running at http://localhost:${PORT}`);
  console.log(`📂 Workspace Root: ${WORKSPACE_DIR}`);
  console.log(`🛠️ Active Native Tools: ${TOOL_DEFINITIONS.map(t => t.function.name).join(', ')}`);
  console.log(`======================================================\n`);

  // Eagerly trigger model background launch
  try {
    console.log('🔄 Initializing local foundation model in background...');
    modelManager.start().catch(err => {
      console.warn('⚠️ Model initialization deferred or encountered warning:', err.message);
    });

    // Warm load memory
    console.log('🔥 Warming up memory with workspace files...');
    listAllFiles().then(files => {
      memoryManager.warmLoad(files, WORKSPACE_DIR).then(() => {
        console.log(`✅ Memory warmed up with ${files.length} files.`);
      }).catch(err => {
        console.warn('⚠️ Non-fatal memory warmup warning:', err.message);
      });
    }).catch(err => {
      console.warn('⚠️ Non-fatal list files warning:', err.message);
    });
  } catch (e) {
    console.warn('⚠️ Background model start warning:', e.message);
  }
});

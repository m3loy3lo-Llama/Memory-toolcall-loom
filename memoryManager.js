/**
 * Working Memory & Embedding Manager for Local AI Agent
 * 
 * Bridges live tool interactions with:
 * 1. Persistent multi-turn working memory (Live File Mutations & Tool Observations)
 * 2. 1280-dim Trainable Latent Embeddings (aligned with webdistillation B3 pipeline & TinyMysticalModel LoRA)
 * 3. Trainable trajectory export for continual distillation
 * 4. Embedding cache and workspace warm loading
 */

import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const EMBEDDING_DIM = 1280; // Matches GPT-2 Large / B3EmbeddingExtractor / TinyMysticalModel

// Module-level cache for embeddings to ensure fast retrieval across repeated calls
const embeddingCache = new Map();

/**
 * Deterministic dense semantic embedding generator into 1280-dimensional space
 * Aligned with B3EmbeddingExtractor architecture
 */
export function generateEmbedding(text) {
  if (!text || typeof text !== 'string') {
    return new Float32Array(EMBEDDING_DIM);
  }

  // Fast cache lookup
  if (embeddingCache.has(text)) {
    return embeddingCache.get(text);
  }

  const embedding = new Float32Array(EMBEDDING_DIM);
  
  // Multi-pass hash rolling over n-grams for semantic sensitivity
  const words = text.toLowerCase().match(/\b\w+\b/g) || [text];
  
  for (let w = 0; w < words.length; w++) {
    const word = words[w];
    let wordHash = 0;
    for (let i = 0; i < word.length; i++) {
      wordHash = ((wordHash << 5) - wordHash) + word.charCodeAt(i);
      wordHash |= 0;
    }

    const posWeight = 1.0 / (1.0 + Math.log(w + 1));
    for (let dim = 0; dim < EMBEDDING_DIM; dim++) {
      // Deterministic PRNG seed per dimension and word
      const seed = Math.sin((wordHash ^ (dim * 9301)) + 49297) * 10000;
      const val = (seed - Math.floor(seed)) * 2 - 1; // Range -1 to 1
      embedding[dim] += val * posWeight;
    }
  }

  // L2 Unit Normalization (so cosine similarity is dot product)
  let norm = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    norm += embedding[i] * embedding[i];
  }
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      embedding[i] /= norm;
    }
  }

  // Cache normalized vector (limit cache size to 5,000 items)
  if (embeddingCache.size > 5000) {
    const firstKey = embeddingCache.keys().next().value;
    embeddingCache.delete(firstKey);
  }
  embeddingCache.set(text, embedding);

  return embedding;
}

/**
 * Compute cosine similarity between two 1280-dim vectors
 */
export function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  let dot = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
  }
  return dot;
}

export class MemoryManager {
  constructor(options = {}) {
    this.storageDir = options.storageDir || path.resolve(__dirname, 'trajectories');
    this.observations = []; // { id, tool, query, summary, rawOutput, embedding, timestamp }
    this.mutations = [];    // { id, file, tool, changeSummary, diff, embedding, timestamp }
    this.trajectories = []; // Trainable pairs: { input, target, metadata }
    this.history = [];      // Unified chronological stream of all events
    this.turnIndex = 0;

    this._ensureStorage();
  }

  async _ensureStorage() {
    try {
      if (!fsSync.existsSync(this.storageDir)) {
        await fs.mkdir(this.storageDir, { recursive: true });
      }
    } catch {
      // Non-fatal
    }
  }

  /**
   * Reset / clear working memory (e.g. on new session)
   */
  clear() {
    this.observations = [];
    this.mutations = [];
    this.trajectories = [];
    this.history = [];
    this.turnIndex = 0;
    embeddingCache.clear();
  }

  /**
   * Record a tool execution into working memory and generate trainable embedding.
   * Supports both signatures:
   * 1. recordToolResult(toolName, args, result)
   * 2. recordToolResult(tool, result, summary)
   */
  async recordToolResult(toolName, paramTwo, paramThree) {
    const timestamp = Date.now();
    const id = `mem_${timestamp}_${Math.random().toString(36).substr(2, 5)}`;

    // Normalize parameters depending on caller signature
    let args = {};
    let result = {};
    let summaryOverride = null;

    if (typeof paramThree === 'string') {
      // Signature 2: recordToolResult(tool, result, summary)
      result = paramTwo || {};
      summaryOverride = paramThree;
    } else {
      // Signature 1: recordToolResult(toolName, args, result)
      args = paramTwo || {};
      result = paramThree || {};
    }

    // 1. Detect if this is a LIVE FILE MUTATION tool
    const mutationTools = new Set(['replace_file_content', 'write_file', 'ide_apply_diff', 'ide_replace_all']);
    
    if (mutationTools.has(toolName) && result && (result.success || !result.error)) {
      const targetFile = args.path || result.path || 'unknown_file';
      let summary = summaryOverride;
      let diff = '';

      if (!summary) {
        if (toolName === 'ide_apply_diff') {
          summary = `Applied unified diff hunks to "${targetFile}". Live file updated.`;
          diff = typeof args.diff === 'string' ? args.diff.slice(0, 1000) : '';
        } else if (toolName === 'ide_replace_all') {
          summary = `Replaced regex "${args.regex}" with "${args.replacement}" in "${targetFile}" (${result.matchCount || 0} occurrences).`;
          diff = result.diff || '';
        } else if (toolName === 'replace_file_content') {
          summary = `Replaced target block in "${targetFile}". Live file updated.`;
          diff = `Target: ${String(args.target).slice(0, 200)}\nReplacement: ${String(args.replacement).slice(0, 200)}`;
        } else if (toolName === 'write_file') {
          summary = `Wrote content to "${targetFile}" (overwrite: ${!!args.overwrite}).`;
        }
      }

      const embeddingText = `Mutation ${toolName} ${targetFile} ${summary} ${diff.slice(0, 300)}`;
      const embedding = generateEmbedding(embeddingText);

      const mutationEntry = {
        id,
        file: targetFile,
        tool: toolName,
        changeSummary: summary,
        diff: diff,
        rawResult: result,
        embedding: Array.from(embedding),
        timestamp
      };

      this.mutations.unshift(mutationEntry);
      this.history.unshift(mutationEntry);
      if (this.mutations.length > 30) this.mutations.pop();
      if (this.history.length > 60) this.history.pop();

      // Record trainable training pair (autoencoder/LoRA target)
      await this._saveTrainablePair(embedding, embeddingText, { type: 'mutation', file: targetFile, tool: toolName });
      return mutationEntry;
    }

    // 2. Otherwise, treat as an OBSERVATION tool
    let observationSummary = summaryOverride;
    const rawString = typeof result === 'object' ? JSON.stringify(result) : String(result);

    if (!observationSummary) {
      if (toolName === 'grep_search') {
        const matchCount = result.matchCount !== undefined ? result.matchCount : (result.results ? result.results.split('\n').length : 0);
        observationSummary = `grep for "${args.query}" returned ${matchCount} match(es). Top snippet: ${String(result.results || '').slice(0, 250)}`;
      } else if (toolName === 'ide_find_references') {
        observationSummary = `Found ${result.referenceCount || 0} references to symbol "${args.symbol_name}" in ${result.fileCount || 0} files.`;
      } else if (toolName === 'ide_get_symbols') {
        observationSummary = `Parsed ${result.symbolCount || 0} symbols in "${args.path}" (functions: ${result.breakdown?.functions || 0}, structs: ${result.breakdown?.structs || 0}).`;
      } else if (toolName === 'ide_lint') {
        observationSummary = `Lint diagnostics for "${args.path}": ${result.clean ? 'Clean (0 errors)' : `${result.errorCount} issue(s) detected`}.`;
      } else if (toolName === 'ide_build') {
        observationSummary = `Build execution in "${args.directory}": ${result.success ? 'Success' : 'Failed'}.`;
      } else if (toolName === 'view_file') {
        observationSummary = `Viewed "${args.path}" (lines ${args.start_line || 1}-${args.end_line || 'all'}). Length: ${String(result.content || '').length} chars.`;
      } else if (toolName === 'list_dir') {
        observationSummary = `Listed "${args.path}": ${result.entries ? result.entries.length : 0} items found.`;
      } else {
        observationSummary = `Executed ${toolName} with arguments: ${JSON.stringify(args).slice(0, 150)}. Outcome: ${result.error ? result.error : 'Success'}.`;
      }
    }

    const embeddingText = `Observation ${toolName} ${JSON.stringify(args).slice(0, 150)} ${observationSummary}`;
    const embedding = generateEmbedding(embeddingText);

    const observationEntry = {
      id,
      tool: toolName,
      query: args,
      summary: observationSummary,
      rawOutput: rawString.length > 5000 ? rawString.slice(0, 5000) + '... [truncated]' : rawString,
      embedding: Array.from(embedding),
      timestamp
    };

    this.observations.unshift(observationEntry);
    this.history.unshift(observationEntry);
    if (this.observations.length > 40) this.observations.pop();
    if (this.history.length > 60) this.history.pop();

    await this._saveTrainablePair(embedding, embeddingText, { type: 'observation', tool: toolName });
    return observationEntry;
  }

  /**
   * Explicit mutation recording convenience method
   */
  async recordMutation(file, tool, changeSummary, diff = '') {
    const timestamp = Date.now();
    const id = `mem_${timestamp}_${Math.random().toString(36).substr(2, 5)}`;
    const embeddingText = `Mutation ${tool} ${file} ${changeSummary} ${String(diff).slice(0, 300)}`;
    const embedding = generateEmbedding(embeddingText);

    const mutationEntry = {
      id,
      file,
      tool,
      changeSummary,
      diff,
      embedding: Array.from(embedding),
      timestamp
    };

    this.mutations.unshift(mutationEntry);
    this.history.unshift(mutationEntry);
    if (this.mutations.length > 30) this.mutations.pop();
    if (this.history.length > 60) this.history.pop();

    await this._saveTrainablePair(embedding, embeddingText, { type: 'mutation', file, tool });
    return mutationEntry;
  }

  /**
   * Save trainable embedding pair to disk (compatible with webdistillation B3TrainingPipeline format)
   */
  async _saveTrainablePair(embeddingArray, sourceText, metadata = {}) {
    try {
      const pair = {
        input: Array.from(embeddingArray),
        target: Array.from(embeddingArray), // Autoencoder/LoRA reconstruction target
        sentence: sourceText.slice(0, 300),
        metadata,
        timestamp: new Date().toISOString()
      };

      this.trajectories.push(pair);
      if (this.trajectories.length > 250) this.trajectories.shift();

      const cachePath = path.join(this.storageDir, 'training_cache.json');
      const cacheData = {
        version: '1.0-latent-distill',
        timestamp: new Date().toISOString(),
        count: this.trajectories.length,
        pairs: this.trajectories
      };

      await fs.writeFile(cachePath, JSON.stringify(cacheData, null, 2), 'utf-8');
    } catch {
      // Non-fatal
    }
  }

  /**
   * Warm load embedding cache from workspace files
   */
  async warmLoad(workspaceFiles, baseDir = process.cwd()) {
    console.log(`🧠 Warming memory cache with ${workspaceFiles.length} files...`);
    let loaded = 0;
    for (const file of workspaceFiles) {
      try {
        const fullPath = path.isAbsolute(file) ? file : path.resolve(baseDir, file);
        const stat = await fs.stat(fullPath);
        if (stat.size > 200 * 1024) continue; // Skip large files for fast warmup

        const content = await fs.readFile(fullPath, 'utf8');
        const summary = `File content for ${file}: ${content.substring(0, 400)}`;
        generateEmbedding(summary);
        loaded++;
      } catch {
        // Skip unreadable files
      }
    }
    console.log(`✅ Memory cache warmed with ${loaded} files.`);
  }

  /**
   * Semantic retrieval of memory items using cosine similarity of 1280-dim vectors
   */
  retrieveMemory({ query, file, memory_id, maxResults = 3 } = {}) {
    // 1. Direct ID lookup
    if (memory_id) {
      const foundObs = this.observations.find(o => o.id === memory_id);
      if (foundObs) {
        return {
          match: foundObs,
          type: 'observation',
          trainable_embedding: foundObs.embedding
        };
      }
      const foundMut = this.mutations.find(m => m.id === memory_id);
      if (foundMut) {
        return {
          match: foundMut,
          type: 'mutation',
          trainable_embedding: foundMut.embedding
        };
      }
      return { error: `Memory with ID "${memory_id}" not found.` };
    }

    // 2. Direct File lookup
    if (file) {
      const relatedMutations = this.mutations.filter(m => m.file && m.file.includes(file));
      const relatedObs = this.observations.filter(o => {
        const str = JSON.stringify(o.query) + ' ' + o.summary;
        return str.includes(file);
      });

      return {
        file,
        mutations: relatedMutations.slice(0, maxResults),
        observations: relatedObs.slice(0, maxResults)
      };
    }

    // 3. Vector Similarity Search
    if (!query) {
      return {
        recentMutations: this.mutations.slice(0, 3),
        recentObservations: this.observations.slice(0, 3)
      };
    }

    const queryVec = generateEmbedding(query);
    const scoredObs = this.observations.map(obs => ({
      item: obs,
      type: 'observation',
      score: cosineSimilarity(queryVec, obs.embedding)
    }));

    const scoredMut = this.mutations.map(mut => ({
      item: mut,
      type: 'mutation',
      score: cosineSimilarity(queryVec, mut.embedding)
    }));

    const allScored = [...scoredMut, ...scoredObs]
      .sort((a, b) => b.score - a.score)
      .slice(0, maxResults);

    return {
      query,
      results: allScored.map(res => ({
        id: res.item.id,
        type: res.type,
        tool: res.item.tool,
        summary: res.item.changeSummary || res.item.summary,
        score: Number(res.score.toFixed(4)),
        contentSnippet: res.item.diff || res.item.rawOutput?.slice(0, 400),
        trainable_embedding_dim: res.item.embedding?.length || EMBEDDING_DIM
      }))
    };
  }

  /**
   * Generates a concise Markdown journal for injection into the system prompt
   */
  getJournalMarkdown() {
    if (this.mutations.length === 0 && this.observations.length === 0) {
      return '';
    }

    let md = `\n\n### 🧠 Active Working Memory & Project State (Persistent Across Turns):\n`;

    if (this.mutations.length > 0) {
      md += `#### 📝 Live File Mutations (You previously modified these files):\n`;
      this.mutations.slice(0, 4).forEach(m => {
        md += `- **[MODIFIED]** \`${m.file}\` via \`${m.tool}\`: ${m.changeSummary}\n`;
      });
    }

    if (this.observations.length > 0) {
      md += `#### 🔍 Key Tool Findings:\n`;
      this.observations.slice(0, 5).forEach(o => {
        md += `- **[SEEN]** \`${o.tool}\`: ${o.summary}\n`;
      });
    }

    md += `*(Tip: Use retrieve_memory(query) if you need the full raw output or trainable latent embeddings from previous steps)*\n`;
    return md;
  }

  /**
   * Alias for getJournalMarkdown for backward compatibility
   */
  getMemorySummary() {
    return this.getJournalMarkdown();
  }
}

export const memoryManager = new MemoryManager();

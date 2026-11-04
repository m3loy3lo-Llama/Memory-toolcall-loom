import { spawn, exec } from 'child_process';
import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import util from 'util';
import { fileURLToPath } from 'url';

const execPromise = util.promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class ModelManager {
  static findDefaultModel(dir) {
    if (process.env.MODEL_PATH && fsSync.existsSync(process.env.MODEL_PATH)) {
      return path.resolve(process.env.MODEL_PATH);
    }
    const searchDirs = [dir, path.resolve(dir, '../Agent')];
    for (const d of searchDirs) {
      if (fsSync.existsSync(d)) {
        try {
          const files = fsSync.readdirSync(d);
          const gguf = files.find(f => f.endsWith('.gguf') && !f.startsWith('mmproj-'));
          if (gguf) return path.resolve(d, gguf);
        } catch { }
      }
    }
    return path.resolve(dir, 'model.gguf');
  }

  static findMatchingMmproj(modelPath, dir) {
    if (process.env.MMPROJ_PATH && fsSync.existsSync(process.env.MMPROJ_PATH)) {
      return path.resolve(process.env.MMPROJ_PATH);
    }
    if (!modelPath) return null;
    const modelDir = path.dirname(modelPath);
    const modelBase = path.basename(modelPath, '.gguf').toLowerCase();
    const dirsToCheck = [modelDir, dir];

    for (const d of dirsToCheck) {
      if (fsSync.existsSync(d)) {
        try {
          const files = fsSync.readdirSync(d);
          const mmprojs = files.filter(f => f.startsWith('mmproj-') && f.endsWith('.gguf'));
          
          for (const m of mmprojs) {
            const mmBase = m.replace(/^mmproj-/i, '').replace(/\.gguf$/i, '').toLowerCase();
            // Direct substring match
            if (modelBase.includes(mmBase) || mmBase.includes(modelBase)) {
              return path.resolve(d, m);
            }
            // Shared distinctive tokens (e.g. ['gemma', '4', '12b'])
            const modelTokens = modelBase.split(/[-_.]/).filter(t => t.length > 2);
            const mmTokens = mmBase.split(/[-_.]/).filter(t => t.length > 2);
            const shared = modelTokens.filter(t => mmTokens.includes(t));
            if (shared.length >= 2) {
              return path.resolve(d, m);
            }
          }
        } catch { }
      }
    }
    return null;
  }

  constructor(options = {}) {
    this.modelPath = options.modelPath || process.env.MODEL_PATH || ModelManager.findDefaultModel(__dirname);
    this.mmprojPath = options.mmprojPath || process.env.MMPROJ_PATH || ModelManager.findMatchingMmproj(this.modelPath, __dirname);
    this.host = options.host || process.env.LLAMA_HOST || '127.0.0.1';
    this.port = options.port || Number(process.env.LLAMA_PORT) || 8080;
    // Context size: 0 tells llama-server to dynamically use max context size from GGUF metadata
    this.ctxSize = options.ctxSize !== undefined ? Number(options.ctxSize) : (process.env.CTX_SIZE !== undefined ? Number(process.env.CTX_SIZE) : 0);
    this.gpuLayers = options.gpuLayers !== undefined ? Number(options.gpuLayers) : (Number(process.env.GPU_LAYERS) || 99);

    this.process = null;
    this.isStarting = false;
    this.isReady = false;
    this.apiUrl = `http://${this.host}:${this.port}`;

    this._setupProcessHooks();
  }

  _setupProcessHooks() {
    const cleanup = async () => {
      console.log('\n🛑 Shutting down Model Manager...');
      await this.stop();
      process.exit(0);
    };
    process.once('SIGINT', cleanup);
    process.once('SIGTERM', cleanup);
  }

  async killPort(port) {
    try {
      const { stdout } = await execPromise(`lsof -t -i:${port}`);
      const pid = stdout.trim();
      if (pid) {
        console.log(`⚠️ Port ${port} is currently used by PID ${pid}. Terminating process...`);
        await execPromise(`kill -9 ${pid}`);
        await new Promise(r => setTimeout(r, 1200));
      }
    } catch {
      // Port already free
    }
  }

  async isHealthy() {
    try {
      const res = await fetch(`${this.apiUrl}/health`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const body = await res.json().catch(() => ({}));
        return body.status === 'ok' || res.status === 200;
      }
      return false;
    } catch {
      return false;
    }
  }

  /**
   * List all discoverable .gguf models across local folder and adjacent Agent directories
   */
  async listAvailableModels() {
    const searchDirs = [
      __dirname,
      path.resolve(__dirname, '../Agent')
    ];
    const results = [];
    const seen = new Set();

    for (const dir of searchDirs) {
      try {
        if (!fsSync.existsSync(dir)) continue;
        const entries = await fs.readdir(dir);
        const ggufs = entries.filter(f => f.toLowerCase().endsWith('.gguf') && !f.toLowerCase().startsWith('mmproj-'));
        const mmprojs = entries.filter(f => f.toLowerCase().endsWith('.gguf') && f.toLowerCase().startsWith('mmproj-'));

        for (const file of ggufs) {
          const fullPath = path.resolve(dir, file);
          if (seen.has(fullPath)) continue;
          seen.add(fullPath);

          const stat = await fs.stat(fullPath);
          const baseName = file.replace(/\.gguf$/i, '').toLowerCase();
          
          // Strict family matching for mmproj - never fall back to an arbitrary projector
          const matchingMmproj = mmprojs.find(m => {
            const mmBase = m.replace(/^mmproj-/i, '').replace(/\.gguf$/i, '').toLowerCase();
            if (baseName.includes(mmBase) || mmBase.includes(baseName)) return true;
            const modelTokens = baseName.split(/[-_.]/).filter(t => t.length > 2);
            const mmTokens = mmBase.split(/[-_.]/).filter(t => t.length > 2);
            const shared = modelTokens.filter(t => mmTokens.includes(t));
            return shared.length >= 2;
          }) || null;

          results.push({
            name: file,
            path: fullPath,
            directory: dir,
            sizeBytes: stat.size,
            sizeFormatted: (stat.size / (1024 * 1024 * 1024)).toFixed(2) + ' GB',
            hasVision: !!matchingMmproj,
            mmprojPath: matchingMmproj ? path.resolve(dir, matchingMmproj) : null,
            isActive: this.modelPath === fullPath
          });
        }
      } catch (err) {
        console.warn(`⚠️ Error reading model directory ${dir}:`, err.message);
      }
    }
    return results;
  }

  /**
   * Hot-swap active foundation model dynamically at runtime
   */
  async switchModel({ modelPath, mmprojPath = null, ctxSize, gpuLayers } = {}) {
    if (!modelPath) {
      throw new Error('modelPath is required for hot-swapping.');
    }

    const resolvedPath = path.isAbsolute(modelPath) ? modelPath : path.resolve(__dirname, modelPath);
    if (!fsSync.existsSync(resolvedPath)) {
      throw new Error(`Target model file not found at: ${resolvedPath}`);
    }

    let resolvedMmproj = null;
    if (mmprojPath) {
      resolvedMmproj = path.isAbsolute(mmprojPath) ? mmprojPath : path.resolve(__dirname, mmprojPath);
      if (!fsSync.existsSync(resolvedMmproj)) {
        resolvedMmproj = null;
      }
    } else {
      resolvedMmproj = ModelManager.findMatchingMmproj(resolvedPath, __dirname);
    }

    console.log(`🔄 Hot-swapping model to: "${path.basename(resolvedPath)}" (Vision: ${resolvedMmproj ? path.basename(resolvedMmproj) : 'None - pure text'})`);

    // Gracefully stop active llama-server
    await this.stop();

    this.modelPath = resolvedPath;
    this.mmprojPath = resolvedMmproj;
    if (ctxSize !== undefined) this.ctxSize = Number(ctxSize);
    if (gpuLayers !== undefined) this.gpuLayers = Number(gpuLayers);

    // Launch with new model configuration
    await this.start();

    return this.getStatus();
  }

  async start() {
    if (this.isReady && await this.isHealthy()) {
      console.log('✅ llama-server is already running and healthy.');
      return true;
    }

    if (this.isStarting) {
      console.log('⏳ llama-server is currently starting...');
      return false;
    }

    this.isStarting = true;
    this.isReady = false;

    // Check if another llama-server is already on the port and healthy
    if (await this.isHealthy()) {
      console.log('✅ Connected to existing healthy llama-server instance.');
      this.isReady = true;
      this.isStarting = false;
      return true;
    }

    // Free the port
    await this.killPort(this.port);

    if (!fsSync.existsSync(this.modelPath)) {
      this.isStarting = false;
      throw new Error(`Model file not found at: ${this.modelPath}`);
    }

    const args = [
      '-m', this.modelPath,
      '-c', String(this.ctxSize),
      '-np', '1',
      '--flash-attn', 'auto',
      '-ngl', String(this.gpuLayers),
      '--port', String(this.port),
      '--host', this.host
    ];

    const hasMmproj = Boolean(this.mmprojPath && fsSync.existsSync(this.mmprojPath));
    if (hasMmproj) {
      args.push('--mmproj', this.mmprojPath);
      console.log('📷 Multimodal projector enabled:', path.basename(this.mmprojPath));
    } else {
      console.log('📝 Pure text model mode (no multimodal projector attached)');
    }

    console.log(`🚀 Spawning llama-server: llama-server ${args.join(' ')}`);

    let mmprojError = false;
    let processExited = false;
    let exitCode = null;

    this.process = spawn('llama-server', args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false
    });

    this.process.stdout.on('data', (d) => {
      const line = d.toString().trim();
      if (line && (line.includes('HTTP server listening') || line.includes('ready') || line.includes('error'))) {
        console.log(`[llama-server] ${line}`);
      }
    });

    this.process.stderr.on('data', (d) => {
      const line = d.toString().trim();
      if (line.includes('mismatch between text model') ||
          line.includes('hint: you may be using wrong mmproj') ||
          line.includes('mtmd_init_from_file') ||
          line.includes('failed to load clip model')) {
        mmprojError = true;
      }
      if (line && (line.includes('error') || line.includes('warning') || line.includes('CUDA') || line.includes('Metal'))) {
        console.warn(`[llama-server] ${line}`);
      }
    });

    this.process.on('exit', (code, sig) => {
      processExited = true;
      exitCode = code;
      console.log(`[llama-server] Exited with code ${code}, signal ${sig}`);
      this.isReady = false;
      this.isStarting = false;
    });

    // Poll health endpoint
    const MAX_WAIT_MS = 60000;
    const START_TIME = Date.now();
    console.log(`⏳ Waiting for model "${path.basename(this.modelPath)}" initialization in VRAM...`);

    while (Date.now() - START_TIME < MAX_WAIT_MS) {
      // Automatic Graceful Fallback: If mmproj fails to load or text model mismatch occurs,
      // seamlessly restart llama-server in pure text mode without --mmproj!
      if (hasMmproj && (mmprojError || (processExited && exitCode !== 0))) {
        console.warn(`\n⚠️ Incompatible or invalid mmproj detected for "${path.basename(this.modelPath)}".`);
        console.log('🔄 Gracefully falling back to loading pure text model (omitting --mmproj)...\n');
        this.mmprojPath = null;
        this.isStarting = false;
        await this.stop();
        return await this.start();
      }

      if (processExited && !hasMmproj) {
        this.isStarting = false;
        throw new Error(`llama-server process exited unexpectedly with code ${exitCode}`);
      }

      if (await this.isHealthy()) {
        console.log(`🎉 Model "${path.basename(this.modelPath)}" successfully loaded and ready for inference!`);
        this.isReady = true;
        this.isStarting = false;
        return true;
      }
      await new Promise(r => setTimeout(r, 1000));
    }

    this.isStarting = false;
    throw new Error('llama-server timed out during startup.');
  }

  async stop() {
    if (this.process && !this.process.killed) {
      this.process.kill('SIGTERM');
      await new Promise(r => setTimeout(r, 1000));
      if (!this.process.killed) {
        this.process.kill('SIGKILL');
      }
    }
    await this.killPort(this.port);
    this.isReady = false;
    this.isStarting = false;
  }

  getStatus() {
    return {
      isReady: this.isReady,
      isStarting: this.isStarting,
      model: this.modelPath ? path.basename(this.modelPath) : 'Unknown',
      modelPath: this.modelPath,
      hasVision: Boolean(this.mmprojPath && fsSync.existsSync(this.mmprojPath)),
      port: this.port,
      apiUrl: this.apiUrl,
      ctxSize: this.ctxSize,
      gpuLayers: this.gpuLayers
    };
  }
}

export const modelManager = new ModelManager();

import { TOOL_DEFINITIONS, executeTool } from './tools.js';
import { modelManager } from './modelManager.js';
import { memoryManager } from './memoryManager.js';
import { compactionManager } from './distillationEngine.js';

const SYSTEM_PROMPT = `You are the Lead Architect in a hierarchical local AI engineering system.
You have direct native access to filesystem and terminal tools, plus the ability to delegate implementation tasks to a Senior Developer sub-agent, distill reasoning into cold LoRA weights via smart compaction, and switch contextual memory branches.

Hierarchical System Roles:
1. Lead Architect (You): High-level reasoning, strategy, planning, evaluating heuristics, coordinating sub-agents, and compacting context at logical milestones.
2. Senior Developer (Sub-Agent): Delegated via delegate_subagent to execute concrete file navigations, code inspections, and surgical modifications.

Core Capabilities & Tools:
1. Workspace Exploration: list_dir(path), view_file(path, start_line, end_line), grep_search(query, path, is_regex).
2. Atomic Modification: replace_file_content(path, target, replacement), write_file(path, content, overwrite).
3. Terminal & Execution: run_command(command), generate_image(prompt, filename).
4. Headless IDE Tools: ide_get_symbols(path), ide_find_references(symbol_name, target_dir), ide_get_context(path, line, window), ide_apply_diff(path, diff), ide_replace_all(path, regex, replacement), ide_lint(path), ide_build(directory, command).
5. Tiered Memory & Distillation:
   - retrieve_memory(query, file, memory_id): Recall past observations, mutations, and latent embeddings.
   - compact_context(milestone, reason, decision_path, heuristics): Summarize the "why" (heuristics and decisions) and distill warm context into cold LoRA weight updates. Call this when completing a task phase or when context grows!
   - delegate_subagent(task, instructions, target_files): Delegate a concrete development or inspection sub-goal to the Senior Developer worker.
   - switch_memory_branch(branch_name): Switch or create a project/task memory branch to isolate task-specific weights while retaining global heuristics.

Workflow Instructions:
- Focus on the "why": Document heuristics, invariants, and architectural rationale.
- Action Chains: Connect your tools logically (e.g. grep -> view -> diff -> lint -> build).
- When a task or milestone is completed, invoke compact_context to distill your learnings into weight updates for subsequent turns.`;

/**
 * Extract tool calls from OpenAI response or fallback inline JSON
 */
function extractToolCalls(message) {
  // 1. Native OpenAI tool_calls
  if (message.tool_calls && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
    return message.tool_calls.map(tc => {
      let parsedArgs = {};
      try {
        parsedArgs = typeof tc.function.arguments === 'string'
          ? JSON.parse(tc.function.arguments)
          : (tc.function.arguments || {});
      } catch {
        parsedArgs = { raw: tc.function.arguments };
      }
      return {
        id: tc.id || `call_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        name: tc.function.name,
        arguments: parsedArgs
      };
    });
  }

  // 2. Fallback: Check if message.content contains an inline JSON tool call
  const content = message.content || '';
  if (content.includes('{') && content.includes('}')) {
    // Look for {"name": "...", "arguments": {...}} or {"action": "...", "action_input": {...}}
    const toolNames = [
      'list_dir', 'view_file', 'grep_search', 'replace_file_content', 'write_file', 'run_command', 'generate_image',
      'ide_get_structure', 'ide_get_symbols', 'ide_find_references', 'ide_get_context', 'ide_apply_diff', 'ide_replace_all', 'ide_lint', 'ide_build',
      'retrieve_memory', 'compact_context', 'delegate_subagent', 'switch_memory_branch'
    ];
    for (const toolName of toolNames) {
      if (content.includes(toolName)) {
        // Try parsing JSON block
        const startIdx = content.indexOf('{');
        const lastIdx = content.lastIndexOf('}');
        if (startIdx !== -1 && lastIdx > startIdx) {
          const jsonSub = content.substring(startIdx, lastIdx + 1);
          try {
            const parsed = JSON.parse(jsonSub);
            const name = parsed.name || parsed.action;
            const args = parsed.arguments || parsed.action_input || parsed.parameters || {};
            if (toolNames.includes(name)) {
              return [{
                id: `inline_${Date.now()}`,
                name: name,
                arguments: typeof args === 'object' ? args : { query: String(args) }
              }];
            }
          } catch {
            // Not valid JSON block, ignore
          }
        }
      }
    }
  }

  return [];
}

/**
 * Extract reasoning / thought process if present
 */
function extractReasoning(text) {
  if (!text) return { reasoning: null, cleanedText: '' };

  const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/i) ||
                     text.match(/<\|think\|>([\s\S]*?)<turn\|>/i) ||
                     text.match(/\[THOUGHT\]([\s\S]*?)\[\/THOUGHT\]/i);

  if (thinkMatch) {
    const reasoning = thinkMatch[1].trim();
    const cleanedText = text.replace(thinkMatch[0], '').trim();
    return { reasoning, cleanedText };
  }

  return { reasoning: null, cleanedText: text };
}

/**
 * Resilient fetch with exponential backoff retry for llama-server
 */
async function fetchWithRetry(url, options, maxRetries = 2, delayMs = 1200) {
  let lastError;
  for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
    try {
      const response = await fetch(url, options);
      if (response.ok) return response;

      const errText = await response.text().catch(() => '');
      if (attempt <= maxRetries && (response.status === 503 || response.status === 502 || response.status === 500)) {
        console.warn(`⚠️ [Attempt ${attempt}] llama-server status ${response.status}. Retrying in ${delayMs * attempt}ms...`);
        await new Promise(r => setTimeout(r, delayMs * attempt));
        continue;
      }
      throw new Error(`llama-server returned status ${response.status}: ${errText}`);
    } catch (err) {
      lastError = err;
      if (attempt <= maxRetries) {
        console.warn(`⚠️ [Attempt ${attempt}] Failed to reach llama-server: ${err.message}. Retrying in ${delayMs * attempt}ms...`);
        await new Promise(r => setTimeout(r, delayMs * attempt));
      }
    }
  }
  throw lastError;
}

/**
 * Format and summarize tool results to prevent context window bloat and prompt timeouts
 */
function formatToolResultForConversation(toolName, result, memId) {
  const rawStr = typeof result === 'object' ? JSON.stringify(result) : String(result);
  const MAX_TOOL_CHARS = 2400;

  if (rawStr.length <= MAX_TOOL_CHARS) {
    return rawStr;
  }

  // Preserve structured head and tail, omitting bloated middle chunks
  const head = rawStr.slice(0, 1400);
  const tail = rawStr.slice(-800);
  const omitted = rawStr.length - 2200;

  return `${head}\n\n... [TRUNCATED: ${omitted} characters omitted for context efficiency. Full raw output preserved in Working Memory (${memId || 'cached'}). Call retrieve_memory if you need specific sections] ...\n\n${tail}`;
}

/**
 * Main Agent Execution Loop
 */
export async function runAgentLoop({ messages, temperature = 0.2, maxSteps = 30 }) {
  // Ensure model is ready
  await modelManager.start();

  // Prepare dynamic context: Working Memory Journal + Compaction Handover + Action Chains
  const journalMd = memoryManager.getJournalMarkdown();
  const handoverMd = compactionManager.generateHandoverMarkdown();
  const chainsMd = compactionManager.actionChainTracker.getRecentChainsMarkdown();
  const dynamicContext = [journalMd, handoverMd, chainsMd].filter(Boolean).join('\n');
  const effectiveSystemPrompt = dynamicContext ? `${SYSTEM_PROMPT}\n${dynamicContext}` : SYSTEM_PROMPT;

  // Find if a context checkpoint exists in incoming messages
  let checkpointIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m && (m.is_checkpoint || (typeof m.content === 'string' && m.content.includes('[Context Checkpoint')))) {
      checkpointIdx = i;
      break;
    }
  }

  // Only feed conversation steps back in from the checkpoint step forward
  const activeMessages = checkpointIdx !== -1 ? messages.slice(checkpointIdx) : messages;
  if (checkpointIdx > 0) {
    console.log(`📍 [Context Checkpoint Enforced] Pruned ${checkpointIdx} prior message(s). Feeding active context from checkpoint forward (${activeMessages.length} msgs).`);
  }

  const conversation = [];
  const hasSystem = activeMessages.some(m => m.role === 'system');

  if (!hasSystem) {
    conversation.push({ role: 'system', content: effectiveSystemPrompt });
    for (const msg of activeMessages) {
      conversation.push({ role: msg.role, content: msg.content });
    }
  } else {
    for (const msg of activeMessages) {
      if (msg.role === 'system') {
        conversation.push({ role: 'system', content: dynamicContext ? `${msg.content}\n${dynamicContext}` : msg.content });
      } else {
        conversation.push({ role: msg.role, content: msg.content });
      }
    }
  }

  const executedTools = [];
  let currentStep = 0;
  let finalAssistantMessage = null;
  let turnCheckpoint = null;

  while (currentStep < maxSteps) {
    currentStep++;

    const requestPayload = {
      model: modelManager.getStatus().model,
      messages: conversation,
      tools: TOOL_DEFINITIONS,
      tool_choice: 'auto',
      temperature: temperature
    };

    let response;
    try {
      response = await fetchWithRetry(`${modelManager.apiUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload)
      });
    } catch (err) {
      throw new Error(`Failed to contact llama-server at ${modelManager.apiUrl}: ${err.message}`);
    }

    const data = await response.json();
    const choice = data.choices && data.choices[0];
    if (!choice || !choice.message) {
      throw new Error('Invalid response structure received from model.');
    }

    const rawMessage = choice.message;
    const toolCalls = extractToolCalls(rawMessage);

    // If no tool calls, model provided its final answer!
    if (toolCalls.length === 0) {
      const { reasoning, cleanedText } = extractReasoning(rawMessage.content || '');
      finalAssistantMessage = {
        role: 'assistant',
        content: cleanedText,
        reasoning_content: reasoning || rawMessage.reasoning_content || null,
        tool_executions: executedTools
      };
      break;
    }

    // Model wants to call tools
    console.log(`⚡ Step ${currentStep}: Model requested ${toolCalls.length} tool(s):`, 
      toolCalls.map(t => `${t.name}(${JSON.stringify(t.arguments)})`).join(', ')
    );

    // Add assistant's tool-call message to conversation
    conversation.push({
      role: 'assistant',
      content: rawMessage.content || '',
      tool_calls: rawMessage.tool_calls || toolCalls.map(tc => ({
        id: tc.id,
        type: 'function',
        function: {
          name: tc.name,
          arguments: JSON.stringify(tc.arguments)
        }
      }))
    });

    // Execute each tool
    for (const toolCall of toolCalls) {
      const startTime = Date.now();
      let result;
      try {
        result = await executeTool(toolCall.name, toolCall.arguments);
      } catch (toolErr) {
        result = { error: `Tool execution exception: ${toolErr.message}` };
      }
      const durationMs = Date.now() - startTime;

      // Record into working memory (live mutation or observation) with 1280-dim trainable embedding
      let recordedMem = null;
      try {
        recordedMem = await memoryManager.recordToolResult(toolCall.name, toolCall.arguments, result);
      } catch (memErr) {
        console.warn('⚠️ Non-fatal working memory recording warning:', memErr.message);
      }

      // Smart summarization for conversation context (prevents context blowout while preserving raw in memory)
      const contextString = formatToolResultForConversation(toolCall.name, result, recordedMem?.id);

      // Record in ActionChainTracker for workflow transition training
      const isSuccess = !result?.error;
      const resultSummary = isSuccess ? (result?.message || 'Success') : `Error: ${result.error}`;
      compactionManager.actionChainTracker.recordAction(
        toolCall.name,
        toolCall.arguments,
        resultSummary,
        isSuccess
      );

      executedTools.push({
        step: currentStep,
        name: toolCall.name,
        arguments: toolCall.arguments,
        resultSummary: result.error ? `Error: ${result.error}` : (result.success ? 'Success' : 'Completed'),
        durationMs,
        result: result
      });

      // Append tool result message
      conversation.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        name: toolCall.name,
        content: contextString
      });

      // Check if tool established a context checkpoint
      if (toolCall.name === 'compact_context' && result?.checkpoint) {
        turnCheckpoint = result.checkpoint;
        console.log(`📍 [Mid-Turn Checkpoint] Sealed context into checkpoint "${turnCheckpoint.id}".`);

        // Mid-turn pruning: keep system prompt and insert checkpoint anchor so subsequent steps are lightweight
        const sysMsg = conversation[0];
        const anchorMsg = {
          role: 'assistant',
          content: `📍 [Context Checkpoint: ${turnCheckpoint.id}]\n• Milestone: ${turnCheckpoint.milestone}\n• Decisions: ${turnCheckpoint.decisionPath}\n• Heuristics: ${(turnCheckpoint.heuristics || []).join('; ')}\n[Prior conversation steps compacted. Continuing turn from checkpoint.]`
        };
        conversation.length = 0;
        conversation.push(sysMsg, anchorMsg);
      }
    }
  }

  // If loop exited due to step limit, dynamically synthesize checkpoint and allow seamless continuation
  if (!finalAssistantMessage) {
    console.log(`⚠️ Reached maxSteps (${maxSteps}). Generating continuation checkpoint...`);
    try {
      conversation.push({
        role: 'user',
        content: `[System Notification: You have reached the turn step limit (${maxSteps} iterations). Please provide a concise progress checkpoint: 1) What was completed, 2) The current state of code/files, and 3) The remaining steps. The user can reply "continue" to proceed.]`
      });

      const summaryResponse = await fetchWithRetry(`${modelManager.apiUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelManager.getStatus().model,
          messages: conversation,
          temperature: temperature
        })
      });

      if (summaryResponse.ok) {
        const sumData = await summaryResponse.json();
        const sumChoice = sumData.choices && sumData.choices[0];
        if (sumChoice && sumChoice.message) {
          const { reasoning, cleanedText } = extractReasoning(sumChoice.message.content || '');
          finalAssistantMessage = {
            role: 'assistant',
            content: cleanedText,
            reasoning_content: reasoning,
            tool_executions: executedTools,
            can_continue: true
          };
        }
      }
    } catch (sumErr) {
      console.warn('⚠️ Dynamic continuation summary fallback:', sumErr.message);
    }

    if (!finalAssistantMessage) {
      finalAssistantMessage = {
        role: 'assistant',
        content: `I have reached the step limit (${maxSteps} iterations) for this turn. All tool modifications and observations are preserved in Working Memory. Reply "continue" to proceed with the remaining steps.`,
        reasoning_content: null,
        tool_executions: executedTools,
        can_continue: true
      };
    }
  }

  if (finalAssistantMessage) {
    finalAssistantMessage.checkpoint = turnCheckpoint;
  }

  return finalAssistantMessage;
}

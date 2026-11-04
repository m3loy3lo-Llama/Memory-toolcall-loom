/**
 * Senior Developer Sub-Agent
 * 
 * Part of the Hierarchical Architecture:
 * - Lead Architect: Primary model for high-level reasoning, strategy, and compaction.
 * - Senior Developer: Sub-agent focused on surgical data navigation, retrieval, and code execution.
 */

import { TOOL_DEFINITIONS, executeTool } from './tools.js';
import { modelManager } from './modelManager.js';
import { memoryManager } from './memoryManager.js';
import { compactionManager } from './distillationEngine.js';

const SUBAGENT_SYSTEM_PROMPT = `You are a Senior Developer sub-agent operating under the Lead Architect.
Your mandate is precision execution: inspect files, parse symbols, apply atomic code diffs, run builds, and report structured findings.

Guidelines:
- Focus on the assigned task without deviating into high-level re-architecting.
- Use view_file, grep_search, and ide_get_symbols to inspect code directly. Never hallucinate.
- Use ide_apply_diff or replace_file_content for atomic code edits.
- Run ide_lint or ide_build after editing to verify correctness.
- When finished, summarize your exact changes, verification results, and any blockers.`;

function extractSubAgentToolCalls(message) {
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
  return [];
}

/**
 * Execute a delegated task using the Senior Developer sub-agent
 */
export async function runSubAgent({
  task,
  instructions = '',
  targetFiles = [],
  maxSteps = 10,
  temperature = 0.1
}) {
  console.log(`\n👷 [Senior Developer Sub-Agent] Task received: "${task}"`);

  // Ensure model server is active
  await modelManager.start();

  // Get distilled context handover from compaction manager & memory journal
  const handoverMd = compactionManager.generateHandoverMarkdown();
  const memoryMd = memoryManager.getJournalMarkdown();
  
  let targetFilesNote = '';
  if (targetFiles && targetFiles.length > 0) {
    targetFilesNote = `\nTarget Files in Scope: ${targetFiles.map(f => `\`${f}\``).join(', ')}`;
  }

  const systemContent = `${SUBAGENT_SYSTEM_PROMPT}
${targetFilesNote}
${handoverMd}
${memoryMd}
${instructions ? `\nArchitect's Specific Instructions:\n${instructions}` : ''}`;

  const conversation = [
    { role: 'system', content: systemContent },
    { role: 'user', content: `Please execute the following development task:\n${task}` }
  ];

  const executedTools = [];
  let currentStep = 0;
  let finalReport = '';

  // Exclude delegation tool from sub-agent to prevent recursion
  const subAgentTools = TOOL_DEFINITIONS.filter(t => t.function.name !== 'delegate_subagent');

  while (currentStep < maxSteps) {
    currentStep++;

    const payload = {
      model: modelManager.getStatus().model,
      messages: conversation,
      tools: subAgentTools,
      tool_choice: 'auto',
      temperature: temperature
    };

    let response;
    try {
      response = await fetch(`${modelManager.apiUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const txt = await response.text();
        throw new Error(`Model API error (${response.status}): ${txt}`);
      }
    } catch (err) {
      throw new Error(`Sub-agent failed to reach model server: ${err.message}`);
    }

    const data = await response.json();
    const choice = data.choices && data.choices[0];
    if (!choice || !choice.message) {
      throw new Error('Invalid response received from model in sub-agent loop.');
    }

    const rawMessage = choice.message;
    const toolCalls = extractSubAgentToolCalls(rawMessage);

    if (toolCalls.length === 0) {
      finalReport = rawMessage.content || 'Task execution completed.';
      break;
    }

    // Sub-agent is executing tools
    console.log(`🔨 [Senior Developer Step ${currentStep}] Executing ${toolCalls.length} tool(s):`,
      toolCalls.map(t => t.name).join(', ')
    );

    conversation.push({
      role: 'assistant',
      content: rawMessage.content || '',
      tool_calls: rawMessage.tool_calls || toolCalls.map(tc => ({
        id: tc.id,
        type: 'function',
        function: { name: tc.name, arguments: JSON.stringify(tc.arguments) }
      }))
    });

    for (const toolCall of toolCalls) {
      const startTime = Date.now();
      let result;
      let error = null;

      try {
        result = await executeTool(toolCall.name, toolCall.arguments);
      } catch (err) {
        error = err.message;
        result = { error: err.message };
      }

      const durationMs = Date.now() - startTime;
      const isSuccess = !error && !result?.error;
      const resultSummary = isSuccess 
        ? (typeof result === 'object' && result.message ? result.message : 'Success')
        : `Error: ${error || result.error}`;

      // Record in ActionChainTracker for connective tissue training
      compactionManager.actionChainTracker.recordAction(
        toolCall.name,
        toolCall.arguments,
        resultSummary,
        isSuccess
      );

      // Record in memory manager
      if (isSuccess && memoryManager) {
        await memoryManager.recordToolResult(toolCall.name, toolCall.arguments, result).catch(() => {});
      }

      executedTools.push({
        name: toolCall.name,
        arguments: toolCall.arguments,
        durationMs,
        resultSummary,
        result
      });

      conversation.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        name: toolCall.name,
        content: typeof result === 'object' ? JSON.stringify(result) : String(result)
      });
    }
  }

  console.log(`✅ [Senior Developer Sub-Agent] Completed task with ${executedTools.length} tool execution(s).`);

  return {
    success: true,
    task,
    summary: finalReport,
    executedToolsCount: executedTools.length,
    executedTools,
    stepsUsed: currentStep
  };
}

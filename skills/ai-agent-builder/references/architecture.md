# Architecture Reference

Complete architecture diagram and component interactions for AI agent systems.

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Input Channels                            │
│  [Telegram] [Discord] [WhatsApp] [CLI] [Slack] [Matrix]     │
└──────────────────────┬───────────────────────────────────────┘
                       │ InboundMessage{Channel, ChatID,
                       │   SenderID, Content, Media[]}
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    Message Bus                                │
│  ┌─────────────────┐  ┌──────────────────┐                   │
│  │ chan Inbound     │  │ chan Outbound     │                   │
│  │ (buffered: 64)  │  │ (buffered: 64)   │                   │
│  └────────┬────────┘  └────────▲─────────┘                   │
│           │ PublishInbound()   │ PublishOutbound()            │
└───────────┼────────────────────┼─────────────────────────────┘
            │                    │
            ▼                    │
┌───────────────────────────────────────────────────────────────┐
│                      Agent Loop                                │
│                                                                │
│  ┌──────────────┐  ┌───────────┐  ┌────────────────────────┐  │
│  │ HookManager  │  │ EventBus  │  │ SteeringQueue          │  │
│  │ Before/After │  │ Subscribe │  │ Mid-turn user input    │  │
│  │ LLM & Tools  │  │ Emit      │  │ max: 10 per scope     │  │
│  └──────────────┘  └───────────┘  └────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ runTurn(ctx, turnState)                                  │  │
│  │  1. buildContext(systemPrompt + history + tools)         │  │
│  │  2. isOverContextBudget() → compress if needed           │  │
│  │  3. fallbackChain.Execute(providers, callLLM)            │  │
│  │  4. parseResponse → text? toolCalls?                     │  │
│  │  5. if toolCalls → registry.Execute(ctx, toolCall)       │  │
│  │  6. checkSteering() → inject user messages               │  │
│  │  7. checkInterrupt() → graceful/hard abort?              │  │
│  │  8. LOOP back to step 2                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌───────────────┐  ┌──────────────────────────────────────┐  │
│  │ Tool Registry │  │ Router (Complexity Classifier)       │  │
│  │ exec, web,    │  │ Light Model ↔ Heavy Model            │  │
│  │ file, spawn.. │  │ Score-based routing (0.0 → 1.0)      │  │
│  └───────────────┘  └──────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ SubTurn Manager                                          │  │
│  │ Semaphore: max 5 concurrent | Depth: max 3 nested        │  │
│  │ Ephemeral sessions | Shared token budget                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│                  Provider Layer (FallbackChain)                 │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌─────────┐  │
│  │ OpenAI  │ │ Anthropic│ │ Gemini │ │ Ollama │ │ 30+more │  │
│  └─────────┘ └──────────┘ └────────┘ └────────┘ └─────────┘  │
│  CooldownTracker per provider+model | Error classification    │
└───────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Message Bus
- **Type**: Buffered Go channels (size 64)
- **Thread safety**: Triple-select pattern (msg, ctx.Done, default)
- **Purpose**: Decouples input channels from agent loop

### Agent Loop
- **Entry**: `Run(ctx)` — infinite select on bus + idle ticker
- **Concurrency**: `sync.Map` for active turn states (multiple sessions)
- **Hot reload**: `ReloadProviderAndConfig()` with write-locked swap

### Turn State
- **Lifecycle**: setup → running → tools → finalizing → completed/aborted
- **Tracking**: iteration count, phase, interrupt flags, restore points
- **Rollback**: Captures history snapshot at turn start, restores on hard abort

### Steering Queue
- **Scope**: Per-session message queues (map[string][]Message)
- **Modes**: "one-at-a-time" (default) or "all" (drain entire queue)
- **Bound**: MaxQueueSize = 10, returns error when full
- **GC**: Cleared references on dequeue to prevent memory leaks

### Hook Manager
- **Types**: BeforeLLM, AfterLLM, BeforeTool, AfterTool
- **Execution**: Timeout-protected goroutines (prevents blocking)
- **Decisions**: Allow, Modify, Deny, Abort (per-hook)

### Event Bus
- **Pattern**: Non-blocking select/default (drops events, never stalls)
- **Events**: TurnStart, TurnEnd, LLMRequest, LLMResponse, ToolExec, Error
- **Subscribers**: Buffered channels, can handle backpressure

### Tool Registry
- **Safety**: `defer recover()` wraps every tool execution
- **Ordering**: Sorted tool names for deterministic KV cache
- **Hidden tools**: TTL-based visibility (discovered tools expire)

### Provider Layer
- **Interface**: `Chat(ctx, ChatInput) (ChatOutput, error)` — 2 methods
- **Fallback**: Error classification → retry/failover/abort
- **Cooldowns**: Per-provider+model cooldown tracking

## Data Flow — Single Turn

```
1. InboundMessage arrives on bus
2. resolveAgent(msg) → which agent handles this?
3. resolveSession(msg) → which conversation?
4. Start drainBusToSteering goroutine
5. newTurnState(agent, session, opts)
6. captureRestorePoint(history)  // for rollback
7. buildContext(systemPrompt + history + tools + userMsg)
8. isOverContextBudget? → compressHistory(turnBoundaries)
9. hooks.BeforeLLM(messages) → may modify/abort
10. fallbackChain.Execute(callLLM)
    ├── Success: parse response
    └── Error: classify → retry? failover? abort?
11. hooks.AfterLLM(response) → may modify
12. If response has tool_calls:
    a. For each tool_call:
       - hooks.BeforeTool(name, args)
       - registry.Execute(ctx, name, args)  // with recover()
       - hooks.AfterTool(name, result)
    b. checkSteering() → inject queued user messages
    c. checkGracefulInterrupt() → inject "stop and summarize"
    d. GOTO step 7 (next iteration)
13. If response is text:
    a. setFinalContent(text)
    b. setPhase(TurnPhaseFinalizing)
    c. publishOutbound(response)
    d. triggerSummarization if history is long
    e. clearActiveTurn()
```

## Data Flow — SubTurn

```
1. Parent agent calls spawn(task, targetAgent)
2. Acquire semaphore slot (max 5 concurrent)
3. Check depth limit (max 3 nested)
4. Create independent context (context.Background + timeout)
5. Create ephemeral session store (in-memory, 50 msg cap)
6. Clone parent's tools WITHOUT spawn/spawn_status
7. Inherit or set token budget
8. runTurn(childCtx, childTurnState)
9. Child runs its own agent loop (same 7-step cycle)
10. Result delivered to parent's pendingResults channel
11. Release semaphore slot
12. Parent sees result at next steering check
```

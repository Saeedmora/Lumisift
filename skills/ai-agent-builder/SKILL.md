---
name: ai-agent-builder
description: Design and build AI agent systems with production-grade patterns for autonomy, resource efficiency, and safety. Use when building agent loops, LLM tool-calling pipelines, multi-agent delegation, context window management, model routing, human-in-the-loop steering, fallback/retry logic, or any AI agent architecture. Covers Go and general agent engineering.
---

# AI Agent Builder

Build production-grade AI agents using proven patterns from the PicoClaw codebase — an ultra-lightweight agent framework running in <10 MB RAM.

## Core Architecture

Every AI agent is a **loop** with 7 steps:

```
① Receive message → ② Build context → ③ Check budget → ④ Call LLM
→ ⑤ Execute tools → ⑥ Check steering → ⑦ Loop or respond
```

## Agent Loop Pattern

The agent loop is an infinite `select` on a message bus channel:

```go
func (al *AgentLoop) Run(ctx context.Context) error {
    for {
        select {
        case <-ctx.Done(): return nil          // graceful shutdown
        case msg := <-al.bus.InboundChan():    // new message
            go al.drainBusToSteering(ctx, scope) // capture mid-turn msgs
            response, _ := al.processMessage(ctx, msg)
            // After turn: process queued steering messages
            for al.pendingSteeringCount(scope) > 0 {
                response, _ = al.Continue(ctx, session, ch, chatID)
            }
            al.PublishResponse(ctx, ch, chatID, response)
        }
    }
}
```

**Key insight:** While `processMessage` runs (which may take minutes), incoming messages drain into a **steering queue** instead of being lost. After the turn ends, queued messages trigger continuation turns.

## Turn State Machine

Each turn has a lifecycle. Track it with a state struct:

```
setup → running → tools → finalizing → completed | aborted
```

Essential fields:
- `phase` — current lifecycle step
- `iteration` — LLM call count (for `max_tool_iterations` guard)
- `gracefulInterrupt` — "finish current tool, then stop"
- `hardAbort` — "cancel HTTP request + all children NOW"
- `restorePointHistory` — snapshot for rollback on failure
- `tokenBudget` — shared counter across parent + child agents

### Two-Level Interrupt System

1. **Graceful**: Sets a flag → agent completes current tool → injects "summarize and stop" message → LLM produces final response
2. **Hard abort**: Cancels the LLM HTTP context + cascades `cancelFunc()` to all child SubTurns → rolls back session history to before the turn

## Steering — Human-in-the-Loop Autonomy

While the agent works, the user can send corrections:

```go
type steeringQueue struct {
    mu     sync.Mutex
    queues map[string][]Message  // scoped per conversation
    mode   SteeringMode          // "one-at-a-time" | "all"
}
const MaxQueueSize = 10  // bounded — prevents memory overflow
```

**Drain pattern**: A goroutine runs alongside `processMessage`, sorting incoming bus messages:
- Same conversation → push to steering queue
- Different conversation → requeue on the bus

After each tool execution, the agent checks the steering queue and injects messages into the LLM context.

## Context Budget Management

Proactively estimate token usage BEFORE calling the LLM:

```go
// Heuristic: ~2.5 chars per token (no tokenizer dependency!)
func estimateMessageTokens(msg Message) int {
    chars := len(msg.Content) + len(msg.ReasoningContent)
    for _, tc := range msg.ToolCalls {
        chars += len(tc.Function.Name) + len(tc.Function.Arguments)
    }
    chars += 12  // per-message overhead
    tokens := chars * 2 / 5
    tokens += len(msg.Media) * 256  // images are expensive
    return tokens
}

// Check: messages + tool definitions + output reserve < context_window
func isOverContextBudget(contextWindow, messages, toolDefs, maxTokens) bool {
    return (msgTokens + toolTokens + maxTokens) > contextWindow
}
```

### Turn-Boundary Compression

When over budget, drop oldest **complete turns** (never split a tool-call sequence):

```go
func parseTurnBoundaries(history []Message) []int {
    var starts []int
    for i, msg := range history {
        if msg.Role == "user" { starts = append(starts, i) }
    }
    return starts  // safe cut points
}
```

## System Prompt Caching (Two-Level)

1. **Local cache**: Avoid rebuilding from 10+ files on every turn. Invalidate via `stat()` mtime checks (microseconds, not file I/O).
2. **LLM KV cache**: Mark static portion as `cache_control: ephemeral` → Anthropic/OpenAI cache the tokenized prefix across requests.

```go
contentBlocks := []ContentBlock{
    {Text: staticPrompt, CacheControl: &CacheControl{Type: "ephemeral"}},
    {Text: dynamicCtx},  // time, session — changes per request
}
```

## Smart Model Routing

Route simple queries to cheap models (50-80% API cost savings):

```go
func (c *RuleClassifier) Score(f Features) float64 {
    if f.HasAttachments { return 1.0 }       // images → heavy
    var score float64
    if f.TokenEstimate > 200 { score += 0.35 }
    if f.CodeBlockCount > 0  { score += 0.40 }
    if f.RecentToolCalls > 3 { score += 0.25 }
    return min(score, 1.0)
    // < 0.35 → cheap model | >= 0.35 → heavy model
}
```

Zero latency, no ML classifier, language-agnostic.

## SubTurn Delegation (Multi-Agent)

Spawn child agents with isolated memory:

```go
func spawnSubTurn(ctx, parentTS, cfg) (*ToolResult, error) {
    // 1. Semaphore — max 5 concurrent children
    parentTS.concurrencySem <- struct{}{}
    // 2. Depth guard — max 3 levels of nesting
    if parentTS.depth >= maxDepth { return ErrTooDeep }
    // 3. Independent context (child survives parent's graceful end)
    childCtx := context.WithTimeout(context.Background(), 5*time.Minute)
    // 4. Ephemeral memory — never pollutes parent
    agent.Sessions = newEphemeralSession()  // in-memory, capped at 50 msgs
    agent.Tools = parentAgent.Tools.Clone() // NO spawn tool (prevents recursion)
    // 5. Shared token budget
    childTS.tokenBudget = parentTS.tokenBudget
}
```

**Critical**: Clone the tool registry BUT exclude `spawn`/`spawn_status` tools → prevents infinite agent recursion.

## Fallback Chain

Classify errors to decide retry vs failover:

| Error Type | Retriable? | Action |
|---|---|---|
| `rate_limit` | ✅ | Wait cooldown, retry or failover |
| `timeout` | ✅ | Retry with next provider |
| `server_error` | ✅ | Retry with backoff |
| `auth` | ❌ | Skip provider permanently |
| `context_overflow` | ❌ | Compress, don't retry |
| `format` | ❌ | Fix request, don't retry |

Providers that fail enter cooldown (tracked per provider+model key).

## Memory System

Zero-RAM persistent memory using markdown files:

```
workspace/memory/
├── MEMORY.md              ← long-term knowledge
└── 202603/
    ├── 20260328.md        ← daily notes
    └── 20260329.md
```

- All writes use **atomic file operations** (write temp → rename)
- Memory content injected into system prompt automatically
- Agent decides what to remember based on conversation

## Security Sandbox

For safe autonomy, enforce defense-in-depth:

1. **Regex deny-list**: 30+ patterns (rm -rf, sudo, fork bomb, curl|sh, git push)
2. **Workspace jail**: Resolve symlinks, check `filepath.Rel()` is local
3. **Channel restriction**: Remote channels (Telegram, Discord) blocked from exec by default
4. **Fail-closed**: Unknown channel → blocked (not allowed)

## Checklist — Building Your Agent

Use this checklist when designing an AI agent system:

- [ ] Define message bus (inbound/outbound channels)
- [ ] Implement agent loop with `select` on bus + context cancellation
- [ ] Add turn state tracking (phase, iteration count, interrupt flags)
- [ ] Implement context budget estimation before LLM calls
- [ ] Add turn-boundary-aware history compression
- [ ] Cache system prompt with mtime-based invalidation
- [ ] Implement tool registry with `defer recover()` for panic safety
- [ ] Add model routing (rule-based complexity scoring)
- [ ] Implement fallback chain with error classification
- [ ] Add steering queue for mid-turn user input
- [ ] Implement SubTurn delegation with semaphore + depth limits
- [ ] Add ephemeral sessions for child agents (capped at 50 messages)
- [ ] Implement shared token budget across parent/child turns
- [ ] Add security sandbox (regex deny-list + workspace jail)
- [ ] Implement graceful + hard interrupt with session rollback
- [ ] Add persistent memory (MEMORY.md + daily notes)
- [ ] Set all queues/buffers to bounded sizes (prevent OOM)
- [ ] Use atomic.Bool/Int64 for thread-safe state (not bare mutex)

## Reference Files

For detailed code examples and patterns, read:
- **[references/architecture.md](references/architecture.md)** — Full architecture diagram, data flow, and component interactions
- **[references/patterns.md](references/patterns.md)** — Go-specific implementation patterns (channels, generics, build tags)
- **[references/resource-saving.md](references/resource-saving.md)** — Complete resource optimization techniques and benchmarks

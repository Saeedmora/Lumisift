# Resource-Saving Techniques

Complete guide to building AI agents that minimize memory, CPU, API costs, and energy usage.

## Token Economy

### Pre-call Budget Estimation

Estimate token usage before spending money:

```go
// Rule: 2.5 chars ≈ 1 token (works for English, code, JSON)
tokens := (chars * 2) / 5

// Don't forget tool definitions — often 2000+ tokens for 20 tools
toolTokens := estimateToolDefsTokens(toolDefs)

// Formula: messages + tools + output_reserve < context_window
total := msgTokens + toolTokens + maxTokens
if total > contextWindow { compress() }
```

**Why this matters**: A failed LLM call (context overflow) still costs money on some providers. Estimating locally costs ~0μs.

### Token Budget Sharing

Parent and child agents share one `atomic.Int64` counter:

```go
tokenBudget := &atomic.Int64{}
tokenBudget.Store(100_000)  // 100k token budget

// After each LLM call
used := response.Usage.TotalTokens
remaining := tokenBudget.Add(-int64(used))
if remaining <= 0 { return ErrBudgetExhausted }
```

**Effect**: Prevents runaway costs when SubTurns spawn SubTurns.

### Model Routing

Route by complexity — simple messages go to cheap models:

| Signal | Weight | Example |
|--------|--------|---------|
| Token estimate > 200 | +0.35 | Long user message |
| Code blocks > 0 | +0.40 | Contains code |
| Recent tool calls > 3 | +0.25 | Active agentic workflow |
| Has attachments | = 1.0 | Images always heavy |

Score < 0.35 → `gpt-4o-mini` (~$0.15/M tokens)
Score ≥ 0.35 → `gpt-4o` (~$5/M tokens)

**Savings**: 50-80% on API costs with zero latency overhead.

## Memory Optimization

### Bounded Queues Everywhere

Every queue has a maximum size to prevent OOM:

| Queue | Max Size | Overflow Strategy |
|-------|----------|-------------------|
| Message Bus (inbound) | 64 | Block sender |
| Message Bus (outbound) | 64 | Block sender |
| Steering Queue | 10 per scope | Return error |
| Event Subscribers | configurable | Drop events |
| Ephemeral Session | 50 messages | Drop oldest |
| Pending SubTurn Results | 16 | Block spawner |

### Ephemeral Sessions

child agents use in-memory sessions that auto-truncate:

```go
func (e *ephemeralSession) AddMessage(role, content string) {
    e.history = append(e.history, Message{Role: role, Content: content})
    if len(e.history) > 50 {
        e.history = e.history[len(e.history)-50:]
    }
}
```

**Effect**: A SubTurn that runs 200 iterations never has more than 50 messages in memory.

### GC-Friendly Patterns

Clear references when dequeuing:

```go
// BAD — old message stays referenced until slice is GC'd
queue = queue[1:]

// GOOD — explicit clear allows GC to reclaim
queue[0] = Message{}  // clear reference
queue = queue[1:]
```

Delete map entries instead of setting to nil:

```go
// BAD
sq.queues[scope] = nil   // key stays in map

// GOOD
delete(sq.queues, scope)  // key removed entirely
```

## I/O Optimization

### System Prompt Cache

Avoid rebuilding from 10+ files on every turn:

| Approach | Cost | Frequency |
|----------|------|-----------|
| Full rebuild | ~5ms (file I/O) | Every turn (bad) |
| mtime check | ~50μs (stat calls) | Every turn (good) |
| Cache hit | ~1μs (string copy) | Most turns (best) |

Implementation: Track `mtime` of all source files (bootstrap, memory, skills). Only rebuild when any file's mtime exceeds the cached timestamp.

### LLM KV Cache Hints

Mark static content for server-side caching:

```go
// Anthropic: per-block cache_control
{Text: staticPrompt, CacheControl: &CacheControl{Type: "ephemeral"}}

// OpenAI: automatic prefix caching (no annotation needed)
// Gemini: automatic context caching with explicit API
```

**Effect**: ~50% cost reduction on cache hits (Anthropic charges half price for cached tokens).

### Atomic Writes

Prevent data corruption on crash/power loss (critical for embedded hardware):

```go
// 1. Write to temp file
os.WriteFile(path + ".tmp", data, 0o600)
// 2. Sync to disk (important for flash storage!)
file.Sync()
// 3. Atomic rename
os.Rename(path + ".tmp", path)
```

## CPU Optimization

### Event-Driven, Not Polling

The agent loop uses `select` with a 100ms idle ticker — not busy-polling:

```go
select {
case <-ctx.Done(): return nil       // shutdown
case msg := <-al.bus.InboundChan(): // work available
    al.processMessage(ctx, msg)
case <-idleTicker.C:               // periodic health check
    if !al.running.Load() { return nil }
}
```

**CPU usage**: Near-zero when idle. No background threads spinning.

### Lock-Free Status Checks

Use `atomic.Bool` for frequently-read flags:

```go
// BAD — contention under load
mu.Lock()
running = false
mu.Unlock()

// GOOD — lock-free, works across goroutines
running.Store(false)
```

### Zero-Allocation Error Classification

Pattern-match on error strings without allocating:

```go
if strings.Contains(errMsg, "429") || strings.Contains(errMsg, "rate") {
    return FailoverRateLimit
}
```

No regex compilation, no JSON parsing — just string contains checks.

## Network Optimization

### Error Classification Prevents Waste

Don't retry non-retriable errors:

```go
failErr := ClassifyError(err)
if !failErr.IsRetriable() {
    return nil, failErr  // auth error → don't try other providers
}
```

**Savings**: A context_overflow error on OpenAI would also fail on Anthropic. Retrying wastes both time and money.

### Cooldown Tracking

Don't hammer a provider that just rejected you:

```go
type CooldownTracker struct {
    mu       sync.RWMutex
    cooloffs map[string]time.Time  // key: provider+model
}

func (ct *CooldownTracker) IsAvailable(key string) bool {
    ct.mu.RLock()
    defer ct.mu.RUnlock()
    return time.Now().After(ct.cooloffs[key])
}
```

## Build-Time Optimization

### Single Binary, No Dependencies

```makefile
CGO_ENABLED=0 go build -ldflags="-s -w" -o picoclaw
```

- `CGO_ENABLED=0`: Pure Go, no C library dependencies
- `-s -w`: Strip debug symbols → 40% smaller binary
- Result: Single ~15MB binary, runs anywhere

### Cross-Platform from One Codebase

```makefile
# Same source, 6 platforms
GOOS=linux GOARCH=arm64 go build    # Raspberry Pi
GOOS=linux GOARCH=mipsle go build   # MIPS routers
GOOS=linux GOARCH=riscv64 go build  # RISC-V boards
GOOS=windows GOARCH=amd64 go build  # Windows
GOOS=darwin GOARCH=arm64 go build   # macOS M-series
```

## Summary — Resource Impact

| Technique | What it Saves | Impact |
|-----------|---------------|--------|
| Token budget estimation | API costs | Prevents wasted 400 errors |
| Model routing | API costs | 50-80% reduction |
| LLM KV cache hints | API costs | ~50% on cache hits |
| Bounded queues | RAM | Prevents OOM (target <10 MB) |
| Ephemeral sessions | RAM | 50-msg cap per child agent |
| GC-friendly dequeue | RAM | No stale references |
| System prompt cache | CPU + I/O | 100x faster (1μs vs 5ms) |
| Event-driven loop | CPU | Near-zero idle usage |
| Atomic operations | CPU | Lock-free state checks |
| Error classification | Network | No wasted retries |
| Cooldown tracking | Network | No provider hammering |
| Single binary | Disk + deploy | 15MB, zero dependencies |

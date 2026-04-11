# Go Implementation Patterns

Reusable Go patterns for building AI agent systems. Each pattern is production-tested in PicoClaw.

## 1. Interface-Driven Provider Abstraction

Define the smallest possible interface. Any LLM (OpenAI, Anthropic, Ollama) implements 2 methods:

```go
type LLMProvider interface {
    Chat(ctx context.Context, input ChatInput) (*ChatOutput, error)
    Models() []string
}

// Optional capabilities via interface composition
type StreamingProvider interface {
    LLMProvider
    ChatStream(ctx context.Context, input ChatInput) (<-chan StreamDelta, error)
}
```

Check capability at runtime:
```go
if sp, ok := provider.(StreamingProvider); ok {
    return sp.ChatStream(ctx, input)  // stream if supported
}
return provider.Chat(ctx, input)  // fallback to non-streaming
```

## 2. Buffered Channel as Semaphore

Use a buffered channel to limit concurrency without external libraries:

```go
concurrencySem := make(chan struct{}, 5)  // max 5 concurrent

// Acquire
select {
case concurrencySem <- struct{}{}: // got slot
case <-time.After(timeout): return ErrTimeout
}

// Release (MUST be in defer or after completion)
<-concurrencySem
```

## 3. Non-Blocking Event Emission

Never let observers stall the agent:

```go
func (eb *EventBus) Emit(evt Event) {
    for _, sub := range eb.subscribers {
        select {
        case sub.ch <- evt:  // delivered
        default:             // dropped — subscriber too slow
            atomic.AddInt64(&eb.dropped, 1)
        }
    }
}
```

## 4. Triple-Select for Safe Channel Operations

Always include ctx.Done and a default/timeout:

```go
func publish(ctx context.Context, ch chan<- Msg, msg Msg) error {
    select {
    case ch <- msg:     return nil      // sent
    case <-ctx.Done():  return ctx.Err() // cancelled
    default:            return ErrFull   // non-blocking variant
    }
}
```

## 5. Panic Recovery in Tool Execution

Wrap every tool call to prevent agent crashes:

```go
func (r *ToolRegistry) Execute(ctx context.Context, name string, args map[string]any) (result *ToolResult) {
    defer func() {
        if r := recover(); r != nil {
            result = &ToolResult{
                Error: fmt.Sprintf("tool %s panicked: %v", name, r),
            }
        }
    }()
    return tool.Execute(ctx, args)
}
```

## 6. Deterministic Tool Ordering for KV Cache

Sort tool names alphabetically so the tool definitions section of the prompt is identical across requests → enables LLM-side KV cache reuse:

```go
func (r *ToolRegistry) SortedDefinitions() []ToolDefinition {
    names := r.List()
    sort.Strings(names)
    var defs []ToolDefinition
    for _, name := range names {
        if tool, ok := r.Get(name); ok {
            defs = append(defs, tool.Definition())
        }
    }
    return defs
}
```

## 7. Atomic State for Lock-Free Status Checks

Use `atomic.Bool` instead of mutex for frequently-read flags:

```go
type AgentLoop struct {
    running atomic.Bool  // checked every 100ms in idle ticker
}

func (al *AgentLoop) Run(ctx context.Context) error {
    al.running.Store(true)
    // ...
    case <-idleTicker.C:
        if !al.running.Load() { return nil }
}

func (al *AgentLoop) Stop() {
    al.running.Store(false)  // lock-free, instant
}
```

## 8. sync.Map for Concurrent Session Management

Multiple conversations can be active simultaneously:

```go
type AgentLoop struct {
    activeTurnStates sync.Map  // key: sessionKey, value: *turnState
}

func (al *AgentLoop) registerActiveTurn(ts *turnState) {
    al.activeTurnStates.Store(ts.sessionKey, ts)
}

func (al *AgentLoop) getActiveTurnState(sessionKey string) *turnState {
    if val, ok := al.activeTurnStates.Load(sessionKey); ok {
        return val.(*turnState)
    }
    return nil
}
```

## 9. GC-Friendly Queue Dequeue

Clear references when dequeuing to prevent memory leaks in long-running agents:

```go
func dequeue(queue []Message) (Message, []Message) {
    msg := queue[0]
    queue[0] = Message{}  // ← GC can reclaim the old message!
    queue = queue[1:]
    return msg, queue
}
```

## 10. Double-Check Locking for Cache

RWMutex with double-check pattern for thread-safe caching:

```go
func (cb *ContextBuilder) BuildSystemPromptWithCache() string {
    // Fast path: read lock
    cb.mu.RLock()
    if cb.cached != "" && !cb.sourceFilesChanged() {
        result := cb.cached
        cb.mu.RUnlock()
        return result
    }
    cb.mu.RUnlock()

    // Slow path: write lock
    cb.mu.Lock()
    defer cb.mu.Unlock()
    // Double-check (another goroutine may have rebuilt)
    if cb.cached != "" && !cb.sourceFilesChanged() {
        return cb.cached
    }
    cb.cached = cb.BuildSystemPrompt()
    return cb.cached
}
```

## 11. Build Tags for Platform Portability

Same function name, different implementations per OS:

```go
// process_unix.go
//go:build !windows

func prepareCommandForTermination(cmd *exec.Cmd) {
    cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// process_windows.go
//go:build windows

func prepareCommandForTermination(cmd *exec.Cmd) {
    cmd.SysProcAttr = &syscall.SysProcAttr{
        CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
    }
}
```

## 12. Atomic File Writes for Crash Safety

Never write directly to the target file — use temp + rename:

```go
func WriteFileAtomic(path string, data []byte, perm os.FileMode) error {
    tmp := path + ".tmp"
    if err := os.WriteFile(tmp, data, perm); err != nil {
        return err
    }
    if err := os.Rename(tmp, path); err != nil {
        os.Remove(tmp)
        return err
    }
    return nil
}
```

## 13. Error Classification for Smart Retries

Classify errors to avoid wasting API calls:

```go
type FailoverReason string
const (
    FailoverRateLimit  FailoverReason = "rate_limit"    // retriable
    FailoverTimeout    FailoverReason = "timeout"       // retriable
    FailoverAuth       FailoverReason = "auth"          // NOT retriable
    FailoverContext    FailoverReason = "context_overflow" // NOT retriable
)

func ClassifyError(err error) *FailoverError {
    msg := err.Error()
    if strings.Contains(msg, "429") { return &FailoverError{Reason: FailoverRateLimit} }
    if strings.Contains(msg, "401") { return &FailoverError{Reason: FailoverAuth} }
    // ... more patterns
}
```

## 14. Context Propagation via context.Value

Pass turn state through the call chain without parameter pollution:

```go
type turnStateKeyType struct{}
var turnStateKey = turnStateKeyType{}

func withTurnState(ctx context.Context, ts *turnState) context.Context {
    return context.WithValue(ctx, turnStateKey, ts)
}

func turnStateFromContext(ctx context.Context) *turnState {
    ts, _ := ctx.Value(turnStateKey).(*turnState)
    return ts
}
```

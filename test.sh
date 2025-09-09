#!/bin/zsh


# Test mini agent with terminal bench
uv run tb run \
    --dataset terminal-bench-core==head \
    --agent-import-path agents.mini_agent:MiniAgent \
    --task-id hello-world

# Test camel agent with terminal bench
uv run tb run \
    --dataset terminal-bench-core==head \
    --agent-import-path agents.camel_agent:CamelTerminus \
    --task-id hello-world    


uv run tb run \
    --dataset terminal-bench-core==head \
    --agent-import-path agents.camel_terminal_agent:CamelTerminalAgent \
    --task-id hello-world    
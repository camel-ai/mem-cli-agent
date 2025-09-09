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

# Test termius 1 agent with terminal bench
uv run tb run \
    --model-name openai/gpt-4o-mini \
    --dataset terminal-bench-core==head \
    --agent-import-path agents.termius_1:Terminus \
    --task-id hello-world

# # Test termius 2 agent with terminal bench
uv run tb run \
    --model-name openai/gpt-4o-mini \
    --dataset terminal-bench-core==head \
    --agent-import-path agents.termius_2:Terminus2 \
    --task-id hello-world
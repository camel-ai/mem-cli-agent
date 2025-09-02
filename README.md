# Memory-Enabled Agent in CLI Domain

A research project exploring how AI agents can leverage previous task context to improve future performance using Terminal-Bench and CAMEL-AI frameworks.

## Overview

This project investigates the use of memory and context summarization in terminal-based AI agents. We implement and compare different agent architectures to understand how agents can learn from past experiences and apply that knowledge to new tasks.

## Key Features

- **Context-Aware Agents**: Agents that can load and utilize summaries from previous task executions
- **Terminal-Bench Integration**: Uses Terminal-Bench framework for standardized terminal task evaluation
- **CAMEL-AI Summarization**: Leverages CAMEL's `summarize()` functionality for intelligent context extraction
- **Multiple Agent Implementations**: Compare different approaches (simple OpenAI, CAMEL-based)

## Project Structure

```
mem-cli-agent/
├── agents/                 # Agent implementations
│   ├── mini_agent.py      # Simple OpenAI-based agent
│   └── camel_agent.py     # CAMEL-powered agent with memory
├── test.sh               # Test scripts for agents
├── pyproject.toml        # Project dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore rules
```

## Research Goals

1. **Memory Utilization**: How can agents effectively use previous task summaries to improve performance?
2. **Context Transfer**: What information from past tasks is most valuable for future tasks?
3. **Learning Efficiency**: Do memory-enabled agents show measurable improvement over stateless agents?

## Agents

### MiniAgent
- Simple OpenAI GPT-4o-mini based agent
- Stateless execution
- Baseline for comparison

### CamelTerminus  
- CAMEL-AI powered agent
- Memory-enabled with context summarization
- Can load previous task summaries via `summary_path` parameter
- Automatically generates summaries after task completion

## Usage

### Basic Testing
```bash
# Test agents
./test.sh
```

### With Memory Context
```python
from agents.camel_agent import CamelTerminus

# Agent without previous context
agent = CamelTerminus()

# Agent with previous context
agent = CamelTerminus(summary_path="path/to/previous/summary.md")
```

## Dependencies

- `terminal-bench>=0.2.16` - Terminal task evaluation framework
- `openai>=1.0.0` - OpenAI API client
- `camel-ai` - CAMEL-AI multi-agent framework (from GitHub)

## Installation

```bash
# Clone the repository
git clone https://github.com/camel-ai/mem-cli-agent
cd mem-cli-agent

# Install dependencies
pip install -e .

# Initialize CAMEL submodule
git submodule update --init --recursive
cd camel/
make install-editable
```

## Research Methodology

1. **Baseline Evaluation**: Test agents on standard terminal tasks without memory
2. **Memory Integration**: Enable context loading and measure performance differences  
3. **Context Analysis**: Analyze what types of summaries are most effective
4. **Comparative Study**: Compare memory-enabled vs stateless agent performance

## Contributing

This is a research project exploring agent memory and context utilization. Contributions, ideas, and discussions about agent memory architectures are welcome.

## License

This project is for research purposes. See individual component licenses (Terminal-Bench, CAMEL-AI) for their respective terms.

## Acknowledgments

- [Terminal-Bench](https://github.com/laude-institute/terminal-bench) - Terminal task evaluation framework
- [CAMEL-AI](https://github.com/camel-ai/camel) - Multi-agent conversation framework
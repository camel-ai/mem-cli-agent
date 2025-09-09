import os
from pathlib import Path
from openai import OpenAI
from terminal_bench.agents import BaseAgent
from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.terminal.models import TerminalCommand
from dotenv import load_dotenv
load_dotenv()

class MiniAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        )

    @staticmethod
    def name() -> str:
        return "openai-simple-agent"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        # Get AI response for the task
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that executes terminal commands. Respond with only the command to run, no explanation."},
                {"role": "user", "content": f"Task: {instruction}"}
            ],
            max_tokens=100,
            temperature=0
        )
        
        command = response.choices[0].message.content.strip()
        
        # Execute the command in the terminal session
        terminal_command = TerminalCommand(command=command)
        session.send_command(terminal_command)
        
        # Return result with token usage
        return AgentResult(
            total_input_tokens=response.usage.prompt_tokens,
            total_output_tokens=response.usage.completion_tokens
        )
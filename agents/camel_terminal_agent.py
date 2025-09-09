# CAMEL agent using Terminal Toolkit
import os
import logging
from pathlib import Path
from typing import List, Tuple
from terminal_bench.agents.base_agent import BaseAgent, AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

logger = logging.getLogger(__name__)


class CamelTerminalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "CamelTerminalAgent"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        """Execute a task using the Terminal Bench harness.
        
        Args:
            instruction: The task instruction to execute
            session: TmuxSession object for command execution
            logging_dir: Optional directory for logging
            
        Returns:
            AgentResult with token counts and failure mode
        """

        container_name = session.container.name
        if not container_name:
            raise ValueError("Container name is required for DockerExecutor")
        
        # Set up CAMEL working directory for logs and internal operations
        camel_workdir = logging_dir / "CAMEL_WORKDIR"
        if not camel_workdir.exists():
            camel_workdir.mkdir(parents=True)
        
        absolute_camel_workdir = str(camel_workdir.resolve())
        os.environ["CAMEL_WORKDIR"] = absolute_camel_workdir
        print(f"Set CAMEL_WORKDIR to: {os.environ['CAMEL_WORKDIR']}")
        
        session_logs_dir = logging_dir / "session" / 'logs'
        if not session_logs_dir.exists():
            session_logs_dir.mkdir(parents=True)
        print(f"Session logs directory: {session_logs_dir}")

        from util.agent_factory import developer_agent_factory
        
        # Use Terminal-Bench's Docker container with TerminalToolkit's Docker backend
        terminal_toolkit_kwargs = {
            "working_directory": "/app",  # Work in container's /app directory where tests expect files
            "use_docker_backend": True,   # Enable Docker backend
            "docker_container_name": container_name,  # Use Terminal-Bench's container
            "session_logs_dir": str(session_logs_dir),
            "safe_mode": False,  # Allow more commands in Docker environment
        }
        
        print(f"Using Docker container: {container_name}")
        print(f"Docker working directory: /app")
        model_backend_reason = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4_1,
            model_config_dict={
                "stream": False,
            },
        )

        task_id = 'workforce_task'
        camel_agent = developer_agent_factory(
            model_backend_reason,
            task_id,
            terminal_toolkit_kwargs
        )
        camel_agent.reset()

        usr_msg = f"{instruction}\n"

        # Get response information
        response = camel_agent.step(usr_msg)

        total_input_tokens = response.info['usage']['prompt_tokens']
        total_output_tokens = response.info['usage']['completion_tokens']
        
        memory_list = camel_agent._memory._chat_history_block.storage.memory_list

        def create_timestamped_marker_from_memory(records: List[dict]) -> Tuple[float, str]:
            """Create a timestamped marker from memory records."""
            results = []
            print(f"Total records: {len(records)}")
            for record in records:

                if 'func_name' in record['message'].keys():
                    timestamp = record['timestamp']
                    func_name = record['message']['func_name']
                    args = record['message'].get('args', {})
                    if args:
                        command = args.get('command', '')
                    else:
                        command = ''
                    results.append((timestamp, f"Called tool: {func_name} with args: {command}"))
            return results

        timestamped_markers = create_timestamped_marker_from_memory(memory_list)


        return AgentResult(
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            failure_mode=FailureMode.NONE,
            timestamped_markers=timestamped_markers,
        )

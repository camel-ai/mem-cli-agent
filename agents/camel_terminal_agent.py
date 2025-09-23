# CAMEL agent using Terminal Toolkit
import os
import logging
from pathlib import Path
from typing import List, Tuple
from terminal_bench.agents.base_agent import BaseAgent, AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

from camel.models import ModelFactory
from camel.logger import enable_logging, set_log_file, set_log_level
from camel.types import ModelPlatformType, ModelType


class CamelTerminalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "CamelTerminalAgent"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = "logs",
    ) -> AgentResult:
        """Execute a task using the Terminal Bench harness.

        Args:
            instruction: The task instruction to execute
            session: TmuxSession object for command execution
            logging_dir: Optional directory for logging

        Returns:
            AgentResult with token counts and failure mode
        """
        if logging_dir is None:
            raise ValueError("logging_dir is required for CamelTerminalAgent")

        if not isinstance(logging_dir, Path):
            logging_dir = Path(logging_dir)

        logging_dir.mkdir(parents=True, exist_ok=True)

        camel_logs_dir = logging_dir / "camel"
        camel_logs_dir.mkdir(parents=True, exist_ok=True)

        enable_logging()
        set_log_level(logging.DEBUG)
        set_log_file(str(camel_logs_dir / "camel_chat_agent.log"))

        container_name = session.container.name
        if not container_name:
            raise ValueError("Container name is required for DockerExecutor")

        camel_workdir = logging_dir / "CAMEL_WORKDIR"
        camel_workdir.mkdir(parents=True, exist_ok=True)
        os.environ["CAMEL_WORKDIR"] = str(camel_workdir.resolve())

        session_logs_dir = logging_dir / "session" / "logs"
        session_logs_dir.mkdir(parents=True, exist_ok=True)

        from util.agent_factory import developer_agent_factory

        terminal_toolkit_kwargs = {
            "working_directory": "/app",
            "use_docker_backend": True,
            "docker_container_name": container_name,
            "session_logs_dir": str(session_logs_dir),
            "safe_mode": False,
        }

        model_backend_reason = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4_1,
            model_config_dict={
                "stream": False,
            },
        )

        task_id = "terminal_task"
        camel_agent = developer_agent_factory(
            model_backend_reason,
            task_id,
            terminal_toolkit_kwargs,
        )
        camel_agent.reset()

        usr_msg = f"{instruction}\n"
        response = camel_agent.step(usr_msg)

        total_input_tokens = response.info["usage"]["prompt_tokens"]
        total_output_tokens = response.info["usage"]["completion_tokens"]

        memory_list = camel_agent._memory._chat_history_block.storage.memory_list

        def create_timestamped_marker_from_memory(records: List[dict]) -> Tuple[float, str]:
            """Create a timestamped marker from memory records."""
            results = []
            for record in records:
                if "func_name" in record["message"].keys():
                    timestamp = record["timestamp"]
                    func_name = record["message"]["func_name"]
                    args = record["message"].get("args", {})
                    command = args.get("command", "") if args else ""
                    results.append((timestamp, f"Called tool: {func_name} with args: {command}"))
            return results

        timestamped_markers = create_timestamped_marker_from_memory(memory_list)

        return AgentResult(
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            failure_mode=FailureMode.NONE,
            timestamped_markers=timestamped_markers,
        )
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

MODEL_TYPE = ModelType.GPT_5
DEFAULT_SUMMARY_DIR = Path("agent_summary")
DEFAULT_CONTEXT_FILENAME = "terminal_task.md"


def _find_latest_context_file(
    base_dir: Path, filename: str
) -> Path | None:
    """Return the newest matching context file within ``base_dir`` if any."""

    try:
        entries = list(base_dir.iterdir())
    except FileNotFoundError:
        return None

    newest_path: Path | None = None
    newest_mtime: float = float("-inf")

    for entry in entries:
        if not entry.is_dir():
            continue

        candidate = entry / filename
        if not candidate.exists():
            continue

        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue

        if mtime > newest_mtime:
            newest_mtime = mtime
            newest_path = candidate

    return newest_path


def _resolve_default_context_file() -> Path | None:
    return _find_latest_context_file(DEFAULT_SUMMARY_DIR, DEFAULT_CONTEXT_FILENAME)


DEFAULT_CONTEXT_FILE = _resolve_default_context_file()
with open(Path("prompts/summary_prompt_simple.txt"), "r") as f:
    DEFAULT_SUMMARY_PROMPT = f.read()


class CamelTerminalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "CamelTerminalAgent"

    def __init__(
        self,
        *,
        load_context: bool = False,
        context_working_dir: str | Path | None = None,
        context_filename: str | Path | None = None,
        summary_working_dir: str | Path | None = DEFAULT_SUMMARY_DIR,
        summary_filename: str | None = None,
        task_id: str = "terminal_task",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.load_context = load_context
        self.context_working_dir = (
            Path(context_working_dir)
            if context_working_dir is not None
            else None
        )
        self.context_filename = (
            Path(context_filename) if context_filename is not None else None
        )
        self.summary_working_dir = (
            Path(summary_working_dir)
            if summary_working_dir is not None
            else None
        )
        self.summary_filename = summary_filename or f"{task_id}.md"
        self.task_id = task_id

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = "logs",
    ) -> AgentResult:
        """Execute a task using the Terminal Bench harness."""
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
            "timeout": 120,
        }

        model_backend_reason = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=MODEL_TYPE,
            model_config_dict={
                "stream": False,
            },
        )

        camel_agent = developer_agent_factory(
            model_backend_reason,
            self.task_id,
            terminal_toolkit_kwargs,
        )
        camel_agent.reset()

        if self.load_context:
            if self.context_working_dir is None:
                raise ValueError(
                    "context_working_dir must be provided when load_context is True"
                )
            if self.context_filename is None:
                raise ValueError(
                    "context_filename must be provided when load_context is True"
                )

            from camel.utils.context_utils import ContextUtility

            context_utility = ContextUtility(
                working_directory=str(self.context_working_dir)
            )
            context_path = self.context_filename
            if not context_path.is_absolute():
                context_path = self.context_working_dir / context_path
            context_utility.load_markdown_context_to_memory(
                camel_agent,
                filename=str(context_path),
            )

        usr_msg = f"{instruction}\n"
        response = camel_agent.step(usr_msg)

        summary_kwargs: dict[str, str] = {}
        if self.summary_working_dir is not None:
            summary_kwargs["working_directory"] = str(self.summary_working_dir)
        if self.summary_filename is not None:
            summary_kwargs["filename"] = self.summary_filename
        summary_kwargs["summary_prompt"] = DEFAULT_SUMMARY_PROMPT

        if summary_kwargs:
            camel_agent.summarize(**summary_kwargs)

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


class CamelTerminalAgentWithContext(CamelTerminalAgent):
    @staticmethod
    def name() -> str:
        return "CamelTerminalAgentWithContext"

    def __init__(self, **kwargs) -> None:
        default_context = DEFAULT_CONTEXT_FILE
        if default_context is None:
            raise FileNotFoundError(
                "No context file found under agent_summary with name "
                f"{DEFAULT_CONTEXT_FILENAME}."
            )
        super().__init__(
            load_context=True,
            context_working_dir=DEFAULT_SUMMARY_DIR,
            context_filename=default_context,
            summary_working_dir=DEFAULT_SUMMARY_DIR,
            summary_filename="terminal_task.md",
            task_id="terminal_task",
            **kwargs,
        )

# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import asyncio
import datetime
import os
import platform
import uuid

from camel.agents.chat_agent import ChatAgent
from camel.logger import get_logger
from camel.messages.base import BaseMessage
from camel.models import BaseModelBackend, ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.toolkits import (
    AgentCommunicationToolkit,
    NoteTakingToolkit,
    TerminalToolkit,
    ToolkitMessageIntegration,
)
from camel.types import ModelPlatformType, ModelType
from camel.utils.commons import api_keys_required

logger = get_logger(__name__)

WORKING_DIRECTORY = os.environ.get("CAMEL_WORKDIR")

print(f"Using working directory: {WORKING_DIRECTORY}")


def send_message_to_user(
    message_title: str,
    message_description: str,
    message_attachment: str = "",
) -> str:
    r"""Use this tool to send a tidy message to the user, including a
    short title, a one-sentence description, and an optional attachment.

    This one-way tool keeps the user informed about your progress,
    decisions, or actions. It does not require a response.
    You should use it to:
    - Announce what you are about to do.
      For example:
      message_title="Starting Task"
      message_description="Searching for papers on GUI Agents."
    - Report the result of an action.
      For example:
      message_title="Search Complete"
      message_description="Found 15 relevant papers."
    - Report a created file.
      For example:
      message_title="File Ready"
      message_description="The report is ready for your review."
      message_attachment="report.pdf"
    - State a decision.
      For example:
      message_title="Next Step"
      message_description="Analyzing the top 10 papers."
    - Give a status update during a long-running task.

    Args:
        message_title (str): The title of the message.
        message_description (str): The short description.
        message_attachment (str): The attachment of the message,
            which can be a file path or a URL.

    Returns:
        str: Confirmation that the message was successfully sent.
    """
    print(f"\nAgent Message:\n{message_title} " f"\n{message_description}\n")
    if message_attachment:
        print(message_attachment)
    logger.info(
        f"\nAgent Message:\n{message_title} "
        f"{message_description} {message_attachment}"
    )
    return (
        f"Message successfully sent to user: '{message_title} "
        f"{message_description} {message_attachment}'"
    )


def developer_agent_factory(
    model: BaseModelBackend,
    task_id: str,
    terminal_toolkit_kwargs: dict = None,
):
    r"""Factory for creating a developer agent."""
    # Initialize message integration
    message_integration = ToolkitMessageIntegration(
        message_handler=send_message_to_user
    )

    # Initialize toolkits
    # terminal_toolkit = TerminalToolkit(safe_mode=True, clone_current_env=False)
    terminal_toolkit = TerminalToolkit(**terminal_toolkit_kwargs)
    note_toolkit = NoteTakingToolkit(working_directory=WORKING_DIRECTORY)

    # Add messaging to toolkits
    terminal_toolkit = message_integration.register_toolkits(terminal_toolkit)
    note_toolkit = message_integration.register_toolkits(note_toolkit)

    # Get enhanced tools
    tools = [
        *terminal_toolkit.get_tools(),
        *note_toolkit.get_tools(),
    ]

    # Determine environment info based on Docker usage
    if terminal_toolkit_kwargs and terminal_toolkit_kwargs.get('use_docker_backend'):
        # Use Docker container environment
        system_info = "Linux (Docker Container)"
        working_dir = terminal_toolkit_kwargs.get('working_directory', '/app')
        env_note = "You are running inside a Docker container. All commands execute within the containerized environment."
    else:
        # Use host system environment
        system_info = f"{platform.system()} ({platform.machine()})"
        working_dir = WORKING_DIRECTORY
        env_note = "You are running on the host system."

    system_message = f"""
<role>
You are a Lead Software Engineer, a master-level coding assistant with a 
powerful terminal. Your role is to solve technical tasks by writing and 
executing code, installing necessary libraries, interacting with the operating 
system, and deploying applications.
</role>

<operating_environment>
- **System**: {system_info}
- **Working Directory**: `{working_dir}`. {env_note}
- **Current Date**: {datetime.date.today()}.
- **IMPORTANT**: When working with files, use paths relative to the working directory above. 
  Do NOT use host system paths like /Users/... when in a Docker container.
</operating_environment>

<instructions>
- When you complete your task, provide a clear summary of what you accomplished.
- Focus on creating files in the correct location as specified by the task.
</instructions>

<capabilities>
- **Code Execution**: You can write and execute code in any language to solve tasks.
- **Terminal Control**: You have access to the terminal and can run command-line tools, 
  manage files, and interact with the OS. Install missing tools with package managers 
  like `pip3`, `uv`, or `apt-get`.
- **File Operations**: {"Create files directly in the working directory using simple paths like './filename' or 'filename'." if terminal_toolkit_kwargs and terminal_toolkit_kwargs.get('use_docker_backend') else "You can access files from any place in the file system."}
- **Verification**: Test and verify your solutions by executing them.
</capabilities>


<approach>
- Take action to solve problems. Don't just suggest solutions—implement them.
- Use the terminal effectively to execute commands and manage files.
- Verify your solutions by testing them.
</approach>
    """

    return ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Developer Agent",
            content=system_message,
        ),
        model=model,
        tools=tools,
        # toolkits_to_register_agent=[screenshot_toolkit],
    )


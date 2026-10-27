# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
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
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
from camel.agents import ChatAgent
from camel.configs import PerplexityConfig
from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType

"""
Please set the following environment variable:
export PERPLEXITY_API_KEY="your_perplexity_api_key_here"

The Perplexity Agent API serves models from several providers. Pass any
model id from https://docs.perplexity.ai/docs/agent-api/models as a string.
"""

model = ModelFactory.create(
    model_platform=ModelPlatformType.PERPLEXITY,
    model_type="openai/gpt-5.5",
    model_config_dict=PerplexityConfig(temperature=0.2).as_dict(),
)

camel_agent = ChatAgent(
    system_message="You are a helpful assistant.", model=model
)
response = camel_agent.step(
    "Say hi to CAMEL AI, one open-source community dedicated to the study "
    "of autonomous and communicative agents."
)
print(response.msgs[0].content)


def add(a: int, b: int) -> int:
    r"""Adds two integers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of the two numbers.
    """
    return a + b


claude_model = ModelFactory.create(
    model_platform=ModelPlatformType.PERPLEXITY,
    model_type="anthropic/claude-sonnet-4-6",
)
tool_agent = ChatAgent(
    system_message="Use the add tool for arithmetic.",
    model=claude_model,
    tools=[FunctionTool(add)],
)
response = tool_agent.step("What is 1234 + 4321?")
print(response.msgs[0].content)

'''
===============================================================================
Hi, CAMEL AI!

Great to meet an open-source community dedicated to advancing the study of
autonomous and communicative agents.
The result of **1234 + 4321 = 5555**!

It's a satisfying answer, all fives!
===============================================================================
'''

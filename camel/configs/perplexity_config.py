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
from __future__ import annotations

from typing import Dict, Optional, Union

from camel.configs.base_config import BaseConfig


class PerplexityConfig(BaseConfig):
    r"""Defines the parameters for generating responses with the Perplexity
    Agent API.

    Reference: https://docs.perplexity.ai/api-reference/agent-post

    Args:
        temperature (float, optional): Sampling temperature to use, between
            :obj:`0` and :obj:`2`. Higher values make the output more random,
            while lower values make it more focused and deterministic.
            (default: :obj:`None`)
        top_p (float, optional): An alternative to sampling with temperature,
            called nucleus sampling, where the model considers the results of
            the tokens with top_p probability mass. (default: :obj:`None`)
        stream (bool, optional): If True, partial response deltas will be
            sent as server-sent events as they become available.
            (default: :obj:`None`)
        max_tokens (int, optional): The maximum number of tokens to generate,
            sent as ``max_output_tokens``. Anthropic models default to
            :obj:`8192` when unset. (default: :obj:`None`)
        tool_choice (Union[dict[str, str], str], optional): Controls which (if
            any) tool is called by the model. :obj:`"none"` means the model
            will not call any tool, :obj:`"auto"` means the model can pick,
            and :obj:`"required"` means the model must call one or more
            tools. (default: :obj:`None`)
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = None
    max_tokens: Optional[int] = None
    tool_choice: Optional[
        Union[Dict[str, Union[str, Dict[str, str]]], str]
    ] = None


PERPLEXITY_API_PARAMS = {
    param for param in PerplexityConfig.model_fields.keys()
}

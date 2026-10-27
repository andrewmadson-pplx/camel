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
import os
from typing import Any, Dict, Optional, Union

from camel.configs import PerplexityConfig
from camel.models.openai_compatible_model import OpenAICompatibleModel
from camel.types import ModelType
from camel.utils import (
    BaseTokenCounter,
    api_keys_required,
)

# The Agent API requires an output token limit for Anthropic models.
_ANTHROPIC_DEFAULT_MAX_TOKENS = 8192


class PerplexityModel(OpenAICompatibleModel):
    r"""LLM API served by the Perplexity Agent API in a unified
    OpenAICompatibleModel interface.

    The Agent API is OpenAI Responses compatible and serves models from
    several providers, such as ``"openai/gpt-5.5"``,
    ``"anthropic/claude-sonnet-5"`` and ``"perplexity/sonar"``.
    Reference: https://docs.perplexity.ai/docs/agent-api/quickstart

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            of request parameters. If :obj:`None`,
            :obj:`PerplexityConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating
            with the Perplexity service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the Perplexity service.
            (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used.
            (default: :obj:`None`)
        timeout (Optional[float], optional): The timeout value in seconds for
            API calls. If not provided, will fall back to the MODEL_TIMEOUT
            environment variable or default to 180 seconds.
            (default: :obj:`None`)
        max_retries (int, optional): Maximum number of retries for API calls.
            (default: :obj:`3`)
        **kwargs (Any): Additional arguments to pass to the client
            initialization.
    """

    @api_keys_required([("api_key", "PERPLEXITY_API_KEY")])
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        from camel import __version__ as camel_version

        if model_config_dict is None:
            model_config_dict = PerplexityConfig().as_dict()
        if (
            str(model_type).startswith("anthropic/")
            and model_config_dict.get("max_tokens") is None
        ):
            model_config_dict = {
                **model_config_dict,
                "max_tokens": _ANTHROPIC_DEFAULT_MAX_TOKENS,
            }
        api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        url = url or os.environ.get(
            "PERPLEXITY_API_BASE_URL", "https://api.perplexity.ai/v1"
        )
        timeout = timeout or float(os.environ.get("MODEL_TIMEOUT", 180))
        kwargs["default_headers"] = {
            "X-Pplx-Integration": f"camel-ai/{camel_version}",
            **(kwargs.get("default_headers") or {}),
        }

        super().__init__(
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key,
            url=url,
            token_counter=token_counter,
            timeout=timeout,
            max_retries=max_retries,
            api_mode="responses",
            **kwargs,
        )

    @staticmethod
    def _responses_chain_enabled(request_config: Dict[str, Any]) -> bool:
        r"""Sends the full conversation on every request. The Agent API
        rejects tool results chained through ``previous_response_id``.
        """
        return False

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
from unittest.mock import patch

import pytest

from camel.configs import PerplexityConfig
from camel.models import ModelFactory, PerplexityModel
from camel.types import ModelPlatformType


@pytest.mark.model_backend
class TestPerplexityModel:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test_key")
        monkeypatch.delenv("PERPLEXITY_API_BASE_URL", raising=False)

    def test_factory_creates_responses_backend(self):
        model = ModelFactory.create(
            model_platform=ModelPlatformType.PERPLEXITY,
            model_type="openai/gpt-5.5",
        )
        assert isinstance(model, PerplexityModel)
        assert model._api_mode == "responses"
        assert model._url == "https://api.perplexity.ai/v1"
        assert model.model_config_dict == PerplexityConfig().as_dict()

    def test_custom_url_from_env(self, monkeypatch):
        monkeypatch.setenv(
            "PERPLEXITY_API_BASE_URL", "https://example.test/v1"
        )
        model = PerplexityModel("openai/gpt-5.5")
        assert model._url == "https://example.test/v1"

    @patch("camel.models.openai_compatible_model.OpenAI")
    def test_integration_header_merges_user_headers(self, mock_openai):
        PerplexityModel("openai/gpt-5.5", default_headers={"X-Custom": "1"})
        headers = mock_openai.call_args.kwargs["default_headers"]
        assert headers["X-Pplx-Integration"].startswith("camel-ai/")
        assert headers["X-Custom"] == "1"

    @pytest.mark.parametrize(
        ("model_type", "config", "expected"),
        [
            ("anthropic/claude-sonnet-4-6", None, 8192),
            ("anthropic/claude-sonnet-4-6", {"max_tokens": 100}, 100),
            ("openai/gpt-5.5", None, None),
        ],
    )
    def test_anthropic_default_max_tokens(self, model_type, config, expected):
        model = PerplexityModel(model_type, model_config_dict=config)
        assert model.model_config_dict.get("max_tokens") == expected

    def test_sends_full_history_instead_of_chaining(self):
        model = PerplexityModel("openai/gpt-5.5")
        assert model._responses_chain_enabled({"store": True}) is False

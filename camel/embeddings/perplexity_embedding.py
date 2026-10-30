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

import base64
import os
from typing import Any

from camel.embeddings.base import BaseEmbedding
from camel.types import EmbeddingModelType
from camel.utils import api_keys_required, dependencies_required


class PerplexityEmbedding(BaseEmbedding[str]):
    r"""Provides text embedding functionalities using the Perplexity
    Embeddings API.

    The API returns base64-encoded signed int8 vectors, which are decoded
    to lists of floats.

    Args:
        model_type (EmbeddingModelType, optional): The model to use for
            embeddings. (default: :obj:`PPLX_EMBED_V1_4B`)
        api_key (Optional[str], optional): The API key for authenticating
            with Perplexity. (default: :obj:`None`)
        dimensions (Optional[int], optional): The dimension of the output
            embeddings. Both models support smaller sizes.
            (default: :obj:`None`)
    """

    @dependencies_required('requests')
    @api_keys_required([("api_key", 'PERPLEXITY_API_KEY')])
    def __init__(
        self,
        model_type: EmbeddingModelType = EmbeddingModelType.PPLX_EMBED_V1_4B,
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        if not model_type.is_perplexity:
            raise ValueError(
                f"Model type {model_type} is not a Perplexity model. "
                "Please use a valid Perplexity model type."
            )
        self.model_type = model_type
        self.dimensions = dimensions
        self.output_dim = dimensions or model_type.output_dim
        self._api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        self.url = 'https://api.perplexity.ai/v1/embeddings'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
        }

    def embed_list(
        self,
        objs: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        r"""Generates embeddings for the given texts.

        Args:
            objs (list[str]): The texts for which to generate the embeddings.
            **kwargs (Any): Extra kwargs passed to the embedding API.

        Returns:
            list[list[float]]: A list that represents the generated embedding
                as a list of floating-point numbers.

        Raises:
            RuntimeError: If the API request fails.
        """
        import requests

        data: dict[str, Any] = {
            "model": self.model_type.value,
            "input": objs,
            "encoding_format": "base64_int8",
            **kwargs,
        }
        if self.dimensions is not None:
            data["dimensions"] = self.dimensions

        try:
            response = requests.post(
                self.url, headers=self.headers, json=data, timeout=180
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to get embeddings from Perplexity: {e}"
            )

        items = sorted(result["data"], key=lambda item: item["index"])
        return [
            [
                float(b - 256 if b > 127 else b)
                for b in base64.b64decode(item["embedding"])
            ]
            for item in items
        ]

    def get_output_dim(self) -> int:
        r"""Returns the output dimension of the embeddings.

        Returns:
            int: The dimensionality of the embedding for the current model.
        """
        return self.output_dim

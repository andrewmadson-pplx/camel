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
import base64
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from camel.embeddings import PerplexityEmbedding
from camel.types import EmbeddingModelType


def _encode(values):
    return base64.b64encode(bytes(v & 0xFF for v in values)).decode()


@patch.dict(os.environ, {"PERPLEXITY_API_KEY": "fake_api_key"})
@patch('requests.post')
def test_embed_list(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"index": 1, "embedding": _encode([3, -4])},
            {"index": 0, "embedding": _encode([0, 127, -128])},
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    embedding = PerplexityEmbedding(
        model_type=EmbeddingModelType.PPLX_EMBED_V1_0_6B, dimensions=256
    )
    result = embedding.embed_list(["first", "second"])

    assert result == [[0.0, 127.0, -128.0], [3.0, -4.0]]
    assert mock_post.call_args[1]["json"] == {
        "model": "pplx-embed-v1-0.6b",
        "input": ["first", "second"],
        "encoding_format": "base64_int8",
        "dimensions": 256,
    }


@patch.dict(os.environ, {"PERPLEXITY_API_KEY": "fake_api_key"})
def test_invalid_model_type():
    with pytest.raises(ValueError, match="is not a Perplexity model"):
        PerplexityEmbedding(model_type=EmbeddingModelType.JINA_EMBEDDINGS_V3)


@patch.dict(os.environ, {"PERPLEXITY_API_KEY": "fake_api_key"})
def test_get_output_dim():
    assert PerplexityEmbedding().get_output_dim() == 2560
    embedding = PerplexityEmbedding(
        model_type=EmbeddingModelType.PPLX_EMBED_V1_0_6B
    )
    assert embedding.get_output_dim() == 1024
    assert PerplexityEmbedding(dimensions=512).get_output_dim() == 512


@patch.dict(os.environ, {"PERPLEXITY_API_KEY": "fake_api_key"})
@patch('requests.post')
def test_api_error_handling(mock_post):
    mock_post.side_effect = requests.exceptions.RequestException("API Error")
    with pytest.raises(RuntimeError, match="Failed to get embeddings"):
        PerplexityEmbedding().embed_list(["test text"])

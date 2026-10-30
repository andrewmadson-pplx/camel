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

from camel.embeddings import PerplexityEmbedding
from camel.types import EmbeddingModelType

# Set the embedding instance
perplexity_embed = PerplexityEmbedding(
    model_type=EmbeddingModelType.PPLX_EMBED_V1_4B,
)

# Embed the text
text_embeddings = perplexity_embed.embed_list(
    ["What is the capital of France?"]
)

print(len(text_embeddings[0]))
'''
===============================================================================
2560
===============================================================================
'''

# Use the smaller model with reduced dimensions
small_embed = PerplexityEmbedding(
    model_type=EmbeddingModelType.PPLX_EMBED_V1_0_6B,
    dimensions=256,
)

small_embeddings = small_embed.embed_list(
    ["What is the capital of France?", "Paris is the capital of France."]
)

print(len(small_embeddings), len(small_embeddings[0]))
'''
===============================================================================
2 256
===============================================================================
'''

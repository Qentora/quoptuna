# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing data generating functions for classification tasks."""

from .bars_and_stripes import generate_bars_and_stripes
from .hidden_manifold import generate_hidden_manifold_model
from .hyperplanes import generate_hyperplanes_parity
from .linearly_separable import generate_linearly_separable
from .two_curves import generate_two_curves

__all__ = [
    "generate_bars_and_stripes",
    "generate_hidden_manifold_model",
    "generate_hyperplanes_parity",
    "generate_linearly_separable",
    "generate_two_curves",
]

# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, esither express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Data utilities."""

import os
import json


def get_dataset_size(source):
    """Return the dataset size."""
    if source.endswith(".json"):
        return len(json.load(open(source, "r", encoding="utf-8")))
    if source.endswith(".txt"):
        return len(open(source, "r").readlines())
    meta_file = os.path.join(source, "METADATA")
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            return json.load(f)["entries"]
    raise ValueError("Unsupported dataset: " + source)

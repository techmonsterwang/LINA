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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""NOVA pipelines."""

from diffnext.pipelines.nova.pipeline_nova import NOVAPipeline
from diffnext.pipelines.nova.pipeline_nova_c2i import NOVAC2IPipeline
from diffnext.pipelines.nova.pipeline_nova_c2i_disa import NOVAC2IPipelineDiSA
from diffnext.pipelines.nova.pipeline_train_c2i import NOVATrainC2IPipeline
from diffnext.pipelines.nova.pipeline_train_t2i import NOVATrainT2IPipeline
from diffnext.pipelines.nova.pipeline_train_t2v import NOVATrainT2VPipeline
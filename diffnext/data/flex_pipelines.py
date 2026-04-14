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
"""Flex data pipelines."""

import multiprocessing

import cv2
import numpy.random as npr

from diffnext.config import cfg
from diffnext.data import flex_transforms
from diffnext.data.builder import LOADERS
from diffnext.data.flex_loaders import DataLoader


class Worker(multiprocessing.Process):
    """Data worker."""

    def __init__(self):
        super(Worker, self).__init__(daemon=True)
        self.seed = cfg.RNG_SEED
        self.reader_queue = None
        self.worker_queue = None

    def run(self):
        """Run implementation."""
        # Disable opencv threading.
        cv2.setNumThreads(1)
        # Fix numpy random seed.
        npr.seed(self.seed)
        # Main loop.
        while True:
            outputs = self.get_outputs(self.reader_queue.get())
            self.worker_queue.put(outputs)


class VAETrainPipe(object):
    """VAE training pipeline."""

    def __init__(self):
        super(VAETrainPipe, self).__init__()
        self.parse_moments = flex_transforms.ParseMoments()
        self.parse_annotations = flex_transforms.ParseAnnotations()

    def get_outputs(self, inputs):
        """Return the outputs."""
        moments = self.parse_moments(inputs)
        label, caption = self.parse_annotations(inputs)
        aspect_ratio = float(moments.shape[-2]) / float(moments.shape[-1])
        outputs = {"moments": [moments], "aspect_ratio": [aspect_ratio]}
        outputs.setdefault("prompt", [label]) if label is not None else None
        outputs.setdefault("prompt", [caption]) if caption is not None else None
        outputs.setdefault("motion_flow", [inputs["flow"]]) if "flow" in inputs else None
        return outputs


class VAETrainWorker(VAETrainPipe, Worker):
    """VAE training worker."""


LOADERS.register("vae_train", DataLoader, worker=VAETrainWorker)

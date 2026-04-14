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
"""Flex data loaders."""

import collections
import multiprocessing as mp
import time
import threading
import queue

import codewithgpu
import numpy as np
import torch

from diffnext.config import cfg
from diffnext.utils import logging


class BalancedQueues(object):
    """Balanced queues."""

    def __init__(self, base_queue, num=1):
        self.queues = [base_queue]
        self.queues += [mp.Queue(base_queue._maxsize) for _ in range(num - 1)]
        self.index = 0

    def put(self, obj, block=True, timeout=None):
        q = self.queues[self.index]
        q.put(obj, block=block, timeout=timeout)
        self.index = (self.index + 1) % len(self.queues)

    def get(self, block=True, timeout=None):
        q = self.queues[self.index]
        obj = q.get(block=block, timeout=timeout)
        self.index = (self.index + 1) % len(self.queues)
        return obj

    def get_n(self, num=1):
        outputs = []
        while len(outputs) < num:
            obj = self.get()
            if obj is not None:
                outputs.append(obj)
        return outputs


class DatasetReader(codewithgpu.DatasetReader):
    """Enhanced dataset reader to apply update."""

    def before_first(self):
        """Move the cursor before begin."""
        self._current = self._first
        self._dataset.seek(self._first)
        self._path2 = self._kwargs.get("path2", "")
        if self._path2 and not hasattr(self, "_dataset2"):
            self._dataset2 = self._dataset_getter(path=self._path2)
        self._dataset2.seek(self._first) if self._path2 else None

    def next_example(self):
        """Return the next example."""
        example = super(DatasetReader, self).next_example()
        example.update(self._dataset2.read()) if self._path2 else None
        return example


class DataLoaderBase(threading.Thread):
    """Base class of data loader."""

    def __init__(self, worker, **kwargs):
        super(DataLoaderBase, self).__init__(daemon=True)
        self.batch_size = kwargs.get("batch_size", 2)
        self.num_readers = kwargs.get("num_readers", 1)
        self.num_workers = kwargs.get("num_workers", 3)
        self.queue_depth = kwargs.get("queue_depth", 2)
        # Initialize distributed group.
        from diffnext.engine import get_ddp_group

        rank, dist_size, dist_group = 0, 1, get_ddp_group()
        if dist_group is not None:
            rank = torch.distributed.get_rank(dist_group)
            dist_size = torch.distributed.get_world_size(dist_group)
        # Build queues.
        self.reader_queue = mp.Queue(self.queue_depth * self.batch_size)
        self.worker_queue = mp.Queue(self.queue_depth * self.batch_size)
        self.batch_queue = queue.Queue(self.queue_depth)
        self.reader_queue = BalancedQueues(self.reader_queue, self.num_workers)
        self.worker_queue = BalancedQueues(self.worker_queue, self.num_workers)
        # Build readers.
        self.readers = []
        for i in range(self.num_readers):
            partition_id = i
            num_partitions = self.num_readers
            num_partitions *= dist_size
            partition_id += rank * self.num_readers
            self.readers.append(
                DatasetReader(
                    output_queue=self.reader_queue,
                    partition_id=partition_id,
                    num_partitions=num_partitions,
                    seed=cfg.RNG_SEED + partition_id,
                    **kwargs,
                )
            )
            self.readers[i].start()
            time.sleep(0.1)
        # Build workers.
        self.workers = []
        for i in range(self.num_workers):
            p = worker()
            p.seed += i + rank * self.num_workers
            p.reader_queue = self.reader_queue.queues[i]
            p.worker_queue = self.worker_queue.queues[i]
            p.start()
            self.workers.append(p)
            time.sleep(0.1)

        # Register cleanup callbacks.
        def cleanup():
            def terminate(processes):
                for p in processes:
                    p.terminate()
                    p.join()

            terminate(self.workers)
            terminate(self.readers)

        import atexit

        atexit.register(cleanup)
        # Start batch prefetching.
        self.start()

    def next(self):
        """Return the next batch of data."""
        return self.__next__()

    def run(self):
        """Main loop."""

    def __call__(self):
        return self.next()

    def __iter__(self):
        """Return the iterator self."""
        return self

    def __next__(self):
        """Return the next batch of data."""
        return [self.batch_queue.get()]


class DataLoader(DataLoaderBase):
    """Loader to return the batch of data."""

    def __init__(self, dataset, worker, **kwargs):
        base_args = {"path": dataset, "path2": kwargs.get("dataset2", None)}
        self.contiguous = kwargs.get("contiguous", True)
        self.prefetch_count = kwargs.get("prefetch_count", 50)
        base_args["shuffle"] = kwargs.get("shuffle", True)
        base_args["batch_size"] = kwargs.get("batch_size", 1)
        base_args["num_workers"] = kwargs.get("num_workers", 1)
        super(DataLoader, self).__init__(worker, **base_args)

    def run(self):
        """Main loop."""
        logging.info("Prefetch batches...")
        next_inputs = []
        prev_inputs = self.worker_queue.get_n(self.prefetch_count * self.batch_size)
        while True:
            # Collect the next batch.
            if len(next_inputs) == 0:
                next_inputs, prev_inputs = prev_inputs, []
            outputs = collections.defaultdict(list)
            for _ in range(self.batch_size):
                inputs = next_inputs.pop(0)
                for k, v in inputs.items():
                    outputs[k].extend(v)
                prev_inputs += self.worker_queue.get_n(1)
            # Stack batch data.
            if self.contiguous:
                outputs["moments"] = np.stack(outputs["moments"])
            # Send batch data to consumer.
            self.batch_queue.put(outputs)

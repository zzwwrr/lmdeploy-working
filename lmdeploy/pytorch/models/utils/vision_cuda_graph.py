# Copyright (c) OpenMMLab. All rights reserved.
"""Bounded, single-stream CUDA graphs for the Intern-Decision vision tower."""
from collections import OrderedDict
from typing import NamedTuple

import torch


class _CapturedShape(NamedTuple):
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor


class VisionCudaGraphs:
    """Reuse launches across different images with the same packed tensor shapes.

    Pixel data, position embeddings and the frame boundaries are runtime inputs.
    Each replay returns its own output before another request can use the pool.
    """

    def __init__(self, max_shapes=16, max_patches=4096, max_frames=16):
        self.max_shapes = max_shapes
        self.max_patches = max_patches
        self.max_frames = max_frames
        self.captured = {}
        self.pending = OrderedDict()
        self.owner = None
        self.capture_stream = None
        self.pool = None
        self.out_of_memory = False

    def clear(self):
        if self.owner is not None:
            self.owner.synchronize()
        self.captured.clear()
        self.pending.clear()
        self.owner = None
        self.capture_stream = None
        self.pool = None
        self.out_of_memory = False

    def _capture(self, eager, args, stream):
        if self.owner is None:
            self.owner = stream
            self.capture_stream = torch.cuda.Stream(device=args[0].device)
            self.pool = torch.cuda.graph_pool_handle()
        static = tuple(value.clone(memory_format=torch.contiguous_format) for value in args)
        self.capture_stream.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.capture_stream, pool=self.pool, capture_error_mode='thread_local'):
            output = eager(static[0], static[1], (static[2], static[3]), static[4])
        stream.wait_stream(self.capture_stream)
        return _CapturedShape(graph, static, output)

    def __call__(self, eager, pixels, ends, rotary, positions):
        count = pixels.size(0)
        frames = ends.numel()
        if (self.out_of_memory or not pixels.is_cuda or not torch.is_inference_mode_enabled()
                or not 0 < count <= self.max_patches
                or not 0 < frames <= self.max_frames or torch.cuda.is_current_stream_capturing()):
            return eager(pixels, ends, rotary, positions)
        stream = torch.cuda.current_stream(pixels.device)
        if self.owner is not None and stream != self.owner:
            return eager(pixels, ends, rotary, positions)
        args = (pixels, ends, *rotary, positions)
        # Strides of the sources may change; capture uses owned contiguous inputs.
        key = tuple((value.shape, value.dtype, value.device) for value in args)
        entry = self.captured.get(key)
        if entry is None:
            if len(self.captured) >= self.max_shapes:
                return eager(pixels, ends, rotary, positions)
            visits = self.pending.pop(key, 0)
            if visits < 2:
                self.pending[key] = visits + 1
                if len(self.pending) > 2 * self.max_shapes:
                    self.pending.popitem(last=False)
                return eager(pixels, ends, rotary, positions)
            try:
                entry = self._capture(eager, args, stream)
            except torch.OutOfMemoryError:
                self.clear()
                self.out_of_memory = True
            if entry is None:
                return eager(pixels, ends, rotary, positions)
            self.captured[key] = entry
        else:
            for dst, src in zip(entry.inputs, args, strict=True):
                dst.copy_(src)
        entry.graph.replay()
        return entry.output.clone()

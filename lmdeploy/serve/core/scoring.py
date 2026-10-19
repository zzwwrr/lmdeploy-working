# Copyright (c) OpenMMLab. All rights reserved.
"""Native prefill scoring; sessions remain owned until confirmed cleanup."""
import asyncio

import torch

from lmdeploy.messages import GenerationConfig, ResponseType


async def score_logits(engine, input_ids, sessions, sequence_start, sequence_end, multimodal, read_positions):
    created = sessions is None
    sessions = sessions if sessions is not None else [engine.session_mgr.get() for _ in input_ids]

    async def score_one(index, session):
        async with session.request_handle(strict=True) as handle:
            generator = None
            submitted = finished = False
            error = None
            try:
                ids = input_ids[index]
                generator = handle.async_stream_infer(
                    session.session_id, input_ids=ids,
                    multimodal=multimodal[index] if multimodal is not None else None,
                    gen_config=GenerationConfig(max_new_tokens=0, output_logits='all', top_k=1,
                                                with_cache=False, preserve_cache=False),
                    sequence_start=sequence_start, sequence_end=sequence_end, step=session.step,
                    stream_output=False)
                submitted = True
                output = None
                async for output in generator:
                    if output.status not in (ResponseType.SUCCESS, ResponseType.FINISH):
                        raise RuntimeError(f'Scoring failed: {output.status}')
                if output is None or output.status != ResponseType.FINISH or output.logits is None:
                    raise RuntimeError('Scoring ended without FINISH and logits')
                finished = True
                full = output.logits
                if full.ndim != 2 or full.shape[0] != len(ids):
                    raise RuntimeError(f'Scoring logits shape {tuple(full.shape)} does not cover {len(ids)} tokens')
                if read_positions is None:
                    result = full
                else:
                    positions = read_positions[index]
                    if not positions or any(isinstance(p, bool) or not isinstance(p, int)
                                            or p < 0 or p >= len(ids) for p in positions):
                        raise ValueError('Scoring positions must be in bounds')
                    vocab = engine.hf_cfg.text_config.vocab_size
                    if full.shape[1] != vocab:
                        raise RuntimeError('Scoring vocabulary dimension mismatch')
                    result = full.index_select(0, torch.tensor(positions, device=full.device)).detach().cpu().clone()
                if not bool(torch.isfinite(result).all()):
                    raise RuntimeError('Scoring returned NaN or Inf')
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                # Implemented by the common session ownership primitive. This runs
                # before request_handle can return the instance to its pool.
                await session.finish_score(handle, generator, submitted=submitted, finished=finished,
                                           end=sequence_end or error is not None,
                                           timeout=getattr(getattr(engine, 'decision_runtime', None), 'config', None),
                                           original_error=error, engine=engine)

    tasks = [asyncio.create_task(score_one(i, session)) for i, session in enumerate(sessions)]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if created and sequence_end:
            for session in sessions:
                engine.session_mgr.remove(session)

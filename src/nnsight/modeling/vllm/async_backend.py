import asyncio
import pickle
import uuid
from typing import TYPE_CHECKING, Any

import zstandard as _zstd

_ZSTD_DECOMPRESSOR = _zstd.ZstdDecompressor()

from ...intervention.backends.base import Backend
from ...intervention.tracing.util import wrap_exception

if TYPE_CHECKING:
    from .vllm import VLLM
else:
    VLLM = Any


class AsyncVLLMBackend(Backend):
    """Backend for async vLLM generation that returns an async generator.

    Usage pattern:
    - ``__call__(tracer)``: Called from ``__exit__``. Compiles the traced
      code, sets up mediators, serializes them into sampling params, and
      immediately submits one request per invoke to the async engine via
      ``.generate()``.
    - ``__aiter__()``: Called by the user via ``async for ... in tracer.backend``.
      Streams ``RequestOutput`` objects from all submitted requests as they
      arrive, attaching collected saves to each finished output.
    """

    def __init__(self, model: "VLLM"):
        self.model = model
        # One ``(request_id, async_generator)`` per invoke. A trace with N
        # invokes submits N requests; the engine runs them concurrently via
        # dynamic batching.
        self._generators = []

    def __call__(self, tracer):
        """Compile traced code, set up mediators, serialize, and submit.

        Uses ``tracer._setup_interleaver()`` directly instead of going
        through ``tracer.execute()`` / ``model.interleave()``, since the
        async path only needs to serialize mediators — not run the model.

        Submits one request per invoke immediately so vLLM can start
        processing them via dynamic batching before the user awaits.
        """
        fn = Backend.__call__(self, tracer)

        try:
            # Set up mediators and collect batched args (shared with sync path).
            args, kwargs = tracer._setup_interleaver(fn)

            if not self.model.dispatched:
                self.model.dispatch()

            # Serialize mediators into sampling params. Each invoke gets its
            # own (prompt, param) pair carrying its mediator.
            prompts, params, lora_requests = self.model._serialize_mediators(
                *args, **kwargs
            )

            # Submit EVERY invoke as its own request. Submitting only
            # prompts[0]/params[0] would serialize the remaining invokes'
            # mediators and then silently drop them, so only prompt 0 would
            # run. Mirrors the fan-out the serve path does (serve/server.py).
            self._generators = []
            for idx, (prompt, param) in enumerate(zip(prompts, params)):
                request_id = str(uuid.uuid4())
                lora_request = lora_requests[idx] if lora_requests else None
                generator = self.model.vllm_entrypoint.generate(
                    prompt, param, request_id, lora_request=lora_request
                )
                self._generators.append((request_id, generator))

            tracer.mediators.clear()
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None

    def __await__(self):
        # Convenience single-await path: drives the first invoke's request.
        # Multi-invoke traces must iterate (``async for``) to stream every
        # invoke's outputs.
        return self._generators[0][1].__await__()

    async def __aiter__(self):
        # Stream outputs from every submitted request concurrently, in arrival
        # order, so all invokes run (not just the first). Each request's
        # generator is drained by its own pump task feeding a shared queue.
        #
        # Saves are collected ONLY on a finished output (one collection per
        # request). Per-invoke saves come back on that invoke's finished
        # output; trace-shared saves are collected by the worker once every
        # invoke's request has finished and ride on the last one to finish.
        queue: asyncio.Queue = asyncio.Queue()

        async def _pump(generator):
            try:
                async for output in generator:
                    queue.put_nowait(("output", output))
            except Exception as e:  # surfaced to the consumer below
                queue.put_nowait(("error", e))
            finally:
                queue.put_nowait(("done", None))

        tasks = [
            asyncio.ensure_future(_pump(generator))
            for _, generator in self._generators
        ]
        pending = len(tasks)

        try:
            while pending > 0:
                kind, payload = await queue.get()
                if kind == "done":
                    pending -= 1
                    continue
                if kind == "error":
                    raise payload
                output = payload
                if output.finished:
                    await self._attach_saves(output)
                yield output
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _attach_saves(self, output) -> None:
        """Collect saves for one finished request and attach to ``output.saves``.

        Surfaces a deferred intervention error (defer mode is always on for
        the vLLM interleaver) before exposing saves, so the user's ``async for``
        raises the original cause instead of silently yielding an output whose
        failed intervention left saves missing. Mirrors the sync path
        (vllm.py) and the serve path. Control-flow signals (tracer.stop())
        are filtered out by ``surface_server_errors``.
        """
        results = await self.model.vllm_entrypoint.collective_rpc(
            "collect_nnsight",
            args=([output.request_id], [output.request_id]),
        )
        saves_bytes = next((r for r in results if r is not None), None)
        if not saves_bytes:
            return

        # Worker returns ``{base_id: {var_name: value}}``. Pull THIS request's
        # sub-dict — the outer layer keeps concurrent independent traces from
        # colliding at shared variable names.
        saves_by_req = pickle.loads(_ZSTD_DECOMPRESSOR.decompress(saves_bytes))
        per_req = saves_by_req.get(output.request_id)
        if not per_req:
            return

        req_exc = per_req.pop("__nnsight_exceptions__", None)
        if req_exc:
            from ...intervention.errors import surface_server_errors

            surface_server_errors(list(req_exc.values()), context="[vLLM]")

        output.saves = per_req

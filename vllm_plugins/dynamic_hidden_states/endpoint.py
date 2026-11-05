"""API-server-side endpoint plugin for aux hidden-state layer swapping.

Registers ``GET``/``POST /aux_hidden_state_layers`` on the OpenAI-compatible
server and forwards to the worker RPC installed by
``AuxLayerWorkerExtension`` via ``EngineClient.collective_rpc`` (a string
method name — raw callables cannot cross the msgpack/ZMQ boundary).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, HTTPException, Request
from starlette.datastructures import State
from vllm.logger import init_logger

if TYPE_CHECKING:
    from argparse import Namespace

    from vllm.engine.protocol import EngineClient

# Namespace under "vllm." so vLLM's logging config surfaces these INFO lines.
logger = init_logger(f"vllm.plugins.{__name__}")

RPC_SET = "set_aux_hidden_state_layers_rpc"
RPC_GET = "get_aux_hidden_state_layers_rpc"
STATE_ATTR = "dynamic_hidden_states_engine_client"


class AuxLayerEndpoint:
    """Endpoint plugin exposing the aux hidden-state layer swap over HTTP."""

    name = "dynamic_hidden_states"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        router = APIRouter()

        @router.get("/aux_hidden_state_layers")
        async def get_layers(request: Request):
            client = _get_client(request)
            results = await client.collective_rpc(RPC_GET)
            return {"layers": list(results[0]) if results else []}

        @router.post("/aux_hidden_state_layers")
        async def set_layers(request: Request, body: dict):
            client = _get_client(request)
            layers = body.get("layers")
            if not isinstance(layers, list) or not layers:
                raise HTTPException(
                    status_code=400,
                    detail="body must be {'layers': [int, ...]} (non-empty)",
                )
            try:
                new = tuple(int(x) for x in layers)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400, detail="all layers must be integers"
                ) from None

            try:
                results = await client.collective_rpc(RPC_SET, args=(new,))
            except Exception as e:  # surfaced to the client as a 400
                # count mismatch, unsupported model, etc. -> client error
                raise HTTPException(status_code=400, detail=str(e)) from e

            applied = list(results[0]) if results else list(new)
            logger.info("Applied aux hidden-state layers via HTTP: %s", applied)
            return {"ok": True, "layers": applied}

        app.include_router(router)

    async def init_state(
        self,
        engine_client: EngineClient | None,
        state: State,
        args: Namespace,  # noqa: ARG002 - required by EndpointPlugin protocol
    ) -> None:
        setattr(state, STATE_ATTR, engine_client)


def _get_client(request: Request) -> EngineClient:
    client = getattr(request.app.state, STATE_ATTR, None)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="engine client unavailable (render server or not initialized)",
        )
    return client

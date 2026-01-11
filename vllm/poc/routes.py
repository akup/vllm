"""PoC (Proof of Compute) API routes for vLLM server.

Implements artifact-based PoC protocol per production-phase-1.md.
"""
import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import aiohttp
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from vllm.logger import init_logger
from .config import PoCState, PoCConfig
from .data import (
    Artifact, Encoding, PoCParams,
    encode_vector, decode_vector,
    compare_artifacts, fraud_test,
    DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD,
)

logger = init_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

# Callback interval for /init/generate (seconds)
POC_CALLBACK_INTERVAL_SEC = float(os.environ.get("POC_CALLBACK_INTERVAL_SEC", "5"))

# Per-chunk timeout for /generate endpoint when engine is busy (seconds)
POC_GENERATE_CHUNK_TIMEOUT_SEC = float(os.environ.get("POC_GENERATE_CHUNK_TIMEOUT_SEC", "60"))

# Module-level state
_poc_tasks: Dict[int, Dict[str, Any]] = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class PoCParamsModel(BaseModel):
    """Strict params for PoC requests."""
    model: str
    seq_len: int
    k_dim: int = 12


class PoCInitGenerateRequest(BaseModel):
    """Request for /init/generate endpoint."""
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    batch_size: int = 32
    params: PoCParamsModel
    url: Optional[str] = None


class ArtifactModel(BaseModel):
    """Single artifact for request/response."""
    nonce: int
    vector_b64: str


class EncodingModel(BaseModel):
    """Encoding metadata."""
    dtype: str = "f16"
    k_dim: int = 12
    endian: str = "le"


class ValidationModel(BaseModel):
    """Artifacts to validate against."""
    artifacts: List[ArtifactModel]


class StatTestModel(BaseModel):
    """Statistical test parameters."""
    dist_threshold: float = DEFAULT_DIST_THRESHOLD
    p_mismatch: float = DEFAULT_P_MISMATCH
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD


class PoCGenerateRequest(BaseModel):
    """Request for /generate endpoint."""
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: List[int]
    params: PoCParamsModel
    batch_size: int = 20
    wait: bool = False
    url: Optional[str] = None
    validation: Optional[ValidationModel] = None
    stat_test: Optional[StatTestModel] = None


# =============================================================================
# Helper Functions
# =============================================================================

async def get_engine_client(request: Request):
    """Get engine client from request app state."""
    engine_client = getattr(request.app.state, 'engine_client', None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="Engine not available")
    return engine_client


async def check_poc_enabled(request: Request):
    """Check if PoC is enabled."""
    poc_enabled = getattr(request.app.state, 'poc_enabled', False)
    if not poc_enabled:
        raise HTTPException(status_code=503, detail="PoC not enabled")


def check_mp_engine_required(engine_client):
    """Check that we're using MP engine for PoC coexistence.
    
    PoC+chat coexistence is only supported in multiprocessing engine mode.
    In-process mode risks NCCL deadlocks from concurrent GPU work.
    """
    from vllm.engine.multiprocessing.client import MQLLMEngineClient
    if not isinstance(engine_client, MQLLMEngineClient):
        raise HTTPException(
            status_code=503,
            detail="PoC coexistence requires multiprocessing engine mode. "
                   "Remove --disable-frontend-multiprocessing or use MP engine."
        )


def check_params_match(request: Request, params: PoCParamsModel):
    """Check if model matches deployed model. Raises 409 if mismatch."""
    # Get model names from openai_serving_models
    serving_models = getattr(request.app.state, 'openai_serving_models', None)
    if not serving_models or not hasattr(serving_models, 'base_model_paths'):
        return  # No model info available
    
    base_paths = serving_models.base_model_paths
    if not base_paths:
        return
    
    model_path = base_paths[0].model_path
    served_names = [p.name for p in base_paths]
    
    # Accept model path OR any served model name
    valid_models = {model_path} | set(served_names)
    if params.model not in valid_models:
        raise HTTPException(
            status_code=409,
            detail=f"model mismatch: requested={params.model}, valid={list(valid_models)}"
        )


async def _cancel_poc_tasks(app_id: int):
    """Cancel running PoC tasks for an app."""
    tasks = _poc_tasks.pop(app_id, None)
    if tasks:
        if tasks.get("stop_event"):
            tasks["stop_event"].set()
        if tasks.get("gen_task"):
            tasks["gen_task"].cancel()
            try:
                await tasks["gen_task"]
            except asyncio.CancelledError:
                pass
        if tasks.get("send_task"):
            tasks["send_task"].cancel()
            try:
                await tasks["send_task"]
            except asyncio.CancelledError:
                pass


# =============================================================================
# Background Tasks for /init/generate
# =============================================================================

# Backoff sleep when PoC is skipped due to chat being busy (seconds)
POC_CHAT_BUSY_BACKOFF_SEC = 0.05

# Timeout for run_batch RPC during coexistence (ms)
# Needs to be longer than VLLM_RPC_TIMEOUT to survive long inference steps
POC_RUN_BATCH_TIMEOUT_MS = int(os.environ.get("POC_RUN_BATCH_TIMEOUT_MS", "60000"))

async def _generation_loop(
    engine_client,
    stop_event: asyncio.Event,
    artifact_queue: asyncio.Queue,
    config: dict,
):
    """Continuous generation loop for /init/generate."""
    total_processed = 0
    start_time = time.time()
    last_report_time = start_time
    
    logger.info(f"PoC generation started")
    
    skip_count = 0
    timeout_count = 0
    try:
        while not stop_event.is_set():
            try:
                # Use longer timeout for run_batch since it waits for engine step
                result = await engine_client.poc_request(
                    "run_batch", {}, timeout_ms=POC_RUN_BATCH_TIMEOUT_MS
                )
                timeout_count = 0  # Reset on successful RPC
            except TimeoutError:
                # Timeout is recoverable - engine is busy with chat inference
                # (long prefill can exceed VLLM_RPC_TIMEOUT)
                timeout_count += 1
                if timeout_count == 1 or timeout_count % 10 == 0:
                    logger.warning(
                        f"PoC run_batch timed out (#{timeout_count}), "
                        "engine busy with inference. Retrying..."
                    )
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)  # Longer backoff for timeout
                continue
            
            if not result.get("should_continue", False):
                logger.info("PoC generation loop ending: should_continue=False")
                break
            
            # Chat-priority: if skipped due to engine step in progress, backoff
            if result.get("skipped"):
                skip_count += 1
                if skip_count % 100 == 1:  # Log every 100 skips (~5s at 50ms backoff)
                    logger.debug(f"PoC yielding to engine step (skip #{skip_count})")
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC)
                continue
            
            skip_count = 0  # Reset on successful batch
            artifacts = result.get("artifacts", [])
            if artifacts:
                await artifact_queue.put({
                    "public_key": result["public_key"],
                    "block_hash": result["block_hash"],
                    "block_height": result["block_height"],
                    "node_id": result["node_id"],
                    "artifacts": artifacts,
                })
            
            total_processed += len(result.get("nonces", []))
            
            # Log progress every 5 seconds
            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                elapsed_min = (current_time - start_time) / 60
                rate = total_processed / elapsed_min if elapsed_min > 0 else 0
                logger.info(f"Generated: {total_processed} nonces in {elapsed_min:.2f}min ({rate:.0f}/min)")
                last_report_time = current_time
            
    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        logger.info(f"PoC stopped: {total_processed} nonces in {elapsed_min:.2f}min")
    except Exception as e:
        # Log all other exceptions so the loop doesn't die silently
        elapsed_min = (time.time() - start_time) / 60
        logger.error(
            f"PoC generation loop crashed after {total_processed} nonces "
            f"in {elapsed_min:.2f}min: {e}",
            exc_info=True
        )
        # Re-raise so the task shows as failed (caller can check)
        raise


async def _callback_sender_loop(
    artifact_queue: asyncio.Queue,
    callback_url: str,
    stop_event: asyncio.Event,
    k_dim: int,
):
    """Batches artifacts and sends callbacks every POC_CALLBACK_INTERVAL_SEC."""
    accumulated: List[dict] = []
    last_send_time = time.time()
    metadata = {}
    
    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                # Wait for artifacts with timeout
                batch = await asyncio.wait_for(
                    artifact_queue.get(),
                    timeout=1.0
                )
                accumulated.extend(batch["artifacts"])
                metadata = {
                    "public_key": batch["public_key"],
                    "block_hash": batch["block_hash"],
                    "block_height": batch["block_height"],
                    "node_id": batch["node_id"],
                }
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break
            
            # Send if interval elapsed and we have data
            current_time = time.time()
            should_send = (
                accumulated and 
                (current_time - last_send_time >= POC_CALLBACK_INTERVAL_SEC or stop_event.is_set())
            )
            
            if should_send:
                payload = {
                    **metadata,
                    "artifacts": [{"nonce": a.nonce, "vector_b64": a.vector_b64} for a in accumulated],
                    "encoding": {"dtype": "f16", "k_dim": k_dim, "endian": "le"},
                }
                
                try:
                    await session.post(
                        f"{callback_url}/generated",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
                    logger.debug(f"Callback sent: {len(accumulated)} artifacts to {callback_url}/generated")
                except Exception as e:
                    logger.warning(f"Callback failed: {e}")
                
                accumulated = []
                last_send_time = current_time
    
    # Send remaining on shutdown
    if accumulated and callback_url:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    **metadata,
                    "artifacts": [{"nonce": a.nonce, "vector_b64": a.vector_b64} for a in accumulated],
                    "encoding": {"dtype": "f16", "k_dim": k_dim, "endian": "le"},
                }
                await session.post(
                    f"{callback_url}/generated",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                )
        except Exception:
            pass


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitGenerateRequest) -> dict:
    """Initialize PoC round and start continuous generation.
    
    Callbacks sent to {url}/generated every POC_CALLBACK_INTERVAL_SEC seconds.
    """
    await check_poc_enabled(request)
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)
    
    # PoC+chat coexistence requires MP engine mode
    check_mp_engine_required(engine_client)
    
    app_id = id(request.app)
    
    # Check for conflicts
    status = await engine_client.poc_request("status", {})
    if status.get("status") == PoCState.GENERATING.value:
        raise HTTPException(status_code=409, detail="Already generating")
    
    # Cancel existing tasks
    await _cancel_poc_tasks(app_id)
    
    # Initialize round
    config = {
        "block_hash": body.block_hash,
        "block_height": body.block_height,
        "public_key": body.public_key,
        "node_id": body.node_id,
        "node_count": body.node_count,
        "batch_size": body.batch_size,
        "seq_len": body.params.seq_len,
        "k_dim": body.params.k_dim,
    }
    
    await engine_client.poc_request("init", config)
    await engine_client.poc_request("start_generate", {})
    
    # Start background tasks
    stop_event = asyncio.Event()
    artifact_queue: asyncio.Queue = asyncio.Queue()
    
    gen_task = asyncio.create_task(
        _generation_loop(engine_client, stop_event, artifact_queue, config)
    )
    
    send_task = None
    if body.url:
        send_task = asyncio.create_task(
            _callback_sender_loop(artifact_queue, body.url, stop_event, body.params.k_dim)
        )
    
    _poc_tasks[app_id] = {
        "gen_task": gen_task,
        "send_task": send_task,
        "stop_event": stop_event,
        "queue": artifact_queue,
        "config": config,
    }
    
    return {
        "status": "OK",
        "pow_status": {"status": "GENERATING"},
    }


@router.post("/generate")
async def generate(request: Request, body: PoCGenerateRequest) -> dict:
    """Compute artifacts for specific nonces. Optionally validate against provided artifacts.
    
    Returns 409 Conflict if /init/generate loop is active.
    """
    await check_poc_enabled(request)
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)
    
    # PoC+chat coexistence requires MP engine mode
    check_mp_engine_required(engine_client)
    
    # Check for conflicts with /init/generate
    status = await engine_client.poc_request("status", {})
    if status.get("status") == PoCState.GENERATING.value:
        raise HTTPException(status_code=409, detail="Busy with /init/generate")
    
    # Validate nonce set match if validation provided
    if body.validation:
        validation_nonces = set(a.nonce for a in body.validation.artifacts)
        request_nonces = set(body.nonces)
        if validation_nonces != request_nonces:
            raise HTTPException(
                status_code=400,
                detail="validation.artifacts nonces must match nonces field exactly"
            )
    
    # If wait=False, just return queued (simplified - no actual queuing for now)
    if not body.wait:
        return {
            "status": "queued",
            "request_id": str(uuid.uuid4()),
            "queued_count": len(body.nonces),
        }
    
    # Compute artifacts in batches (per spec: split nonces into chunks of batch_size)
    # Each chunk retries with backoff if engine is busy (skipped=True)
    computed_artifacts = []
    batch_size = body.batch_size
    
    for i in range(0, len(body.nonces), batch_size):
        chunk = body.nonces[i:i + batch_size]
        chunk_start_time = time.time()
        
        while True:
            result = await engine_client.poc_request("generate_artifacts", {
                "nonces": chunk,
                "block_hash": body.block_hash,
                "public_key": body.public_key,
                "seq_len": body.params.seq_len,
                "k_dim": body.params.k_dim,
            })
            
            # If not skipped, we got our artifacts
            if not result.get("skipped"):
                computed_artifacts.extend(result.get("artifacts", []))
                break
            
            # Check per-chunk timeout
            elapsed = time.time() - chunk_start_time
            if elapsed >= POC_GENERATE_CHUNK_TIMEOUT_SEC:
                raise HTTPException(
                    status_code=503,
                    detail=f"Timeout waiting for engine: chunk {i//batch_size} "
                           f"timed out after {elapsed:.1f}s"
                )
            
            # Backoff before retry (async, doesn't block thread)
            await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC)
    
    # If no validation, return computed artifacts
    if not body.validation:
        return {
            "status": "completed",
            "request_id": str(uuid.uuid4()),
            "artifacts": computed_artifacts,
            "encoding": {"dtype": "f16", "k_dim": body.params.k_dim, "endian": "le"},
        }
    
    # Validation mode: compare computed vs received
    # Build lookup for received artifacts
    received_map = {a.nonce: a.vector_b64 for a in body.validation.artifacts}
    
    # Get stat_test params
    stat_test = body.stat_test or StatTestModel()
    
    # Compare vectors
    n_mismatch = 0
    mismatch_nonces = []
    
    for artifact in computed_artifacts:
        nonce = artifact["nonce"]
        computed_b64 = artifact["vector_b64"]
        received_b64 = received_map.get(nonce)
        
        if received_b64:
            computed_vec = decode_vector(computed_b64)
            received_vec = decode_vector(received_b64)
            distance = np.linalg.norm(computed_vec - received_vec)
            
            if distance > stat_test.dist_threshold:
                n_mismatch += 1
                mismatch_nonces.append(nonce)
    
    n_total = len(body.nonces)
    
    # Run fraud test if stat_test provided
    p_value, fraud_detected = fraud_test(
        n_mismatch, n_total,
        stat_test.p_mismatch, stat_test.fraud_threshold
    )
    
    response = {
        "status": "completed",
        "request_id": str(uuid.uuid4()),
        "n_total": n_total,
        "n_mismatch": n_mismatch,
        "mismatch_nonces": mismatch_nonces,
        "p_value": p_value,
        "fraud_detected": fraud_detected,
    }
    
    # Send callback if URL provided
    if body.url:
        try:
            async with aiohttp.ClientSession() as session:
                callback_payload = {
                    "request_id": response["request_id"],
                    "block_hash": body.block_hash,
                    "block_height": body.block_height,
                    "public_key": body.public_key,
                    "node_id": body.node_id,
                    "n_total": n_total,
                    "n_mismatch": n_mismatch,
                    "mismatch_nonces": mismatch_nonces,
                    "p_value": p_value,
                    "fraud_detected": fraud_detected,
                }
                await session.post(
                    f"{body.url}/validated",
                    json=callback_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                )
        except Exception as e:
            logger.warning(f"Validation callback failed: {e}")
    
    return response


@router.get("/status")
async def get_status(request: Request) -> dict:
    """Get current PoC status."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    return await engine_client.poc_request("status", {})


@router.post("/stop")
async def stop_round(request: Request) -> dict:
    """Stop current PoC round."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Cancel background tasks
    await _cancel_poc_tasks(id(request.app))
    
    result = await engine_client.poc_request("stop", {})
    
    return {
        "status": "OK", 
        "pow_status": result.get("pow_status", {"status": "STOPPED"}),
    }

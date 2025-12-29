"""PoC (Proof of Compute) API routes for vLLM server."""
import asyncio
from typing import List, Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from .config import PoCState
from .data import ProofBatch

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])


class PoCInitRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    r_target: float
    fraud_threshold: float = 0.01
    node_id: int = -1
    node_count: int = -1
    batch_size: int = 32
    seq_len: int = 256
    callback_url: Optional[str] = None


class PoCStatusResponse(BaseModel):
    state: str
    valid_nonces: List[int]
    valid_distances: List[float]
    total_checked: int
    total_valid: int
    elapsed_seconds: float
    rate_per_second: float


class PoCValidateRequest(BaseModel):
    """Request to validate nonces - accepts full ProofBatch format."""
    public_key: str
    block_hash: str
    block_height: int
    nonces: List[int]
    dist: List[float]
    node_id: int


class PoCValidateResponse(BaseModel):
    nonces: List[int]
    distances: List[float]
    valid: List[bool]


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


@router.post("/init")
async def init_round(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round without starting generation."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    result = await engine_client.poc_request("init", body.model_dump())
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round and start generating."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    if body.node_id == -1 or body.node_count == -1:
        raise HTTPException(
            status_code=400, 
            detail="Node ID and node count must be set"
        )
    
    # Initialize
    await engine_client.poc_request("init", body.model_dump())
    
    # Start generating
    result = await engine_client.poc_request("start_generate", {})
    
    # Start background generation loop
    task = asyncio.create_task(_generation_loop(engine_client))
    
    # Store task reference for cleanup
    if not hasattr(request.app.state, '_poc_generation_task'):
        request.app.state._poc_generation_task = None
    if request.app.state._poc_generation_task:
        request.app.state._poc_generation_task.cancel()
    request.app.state._poc_generation_task = task
    
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.post("/init/validate")
async def init_validate(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round and start validating."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Initialize
    await engine_client.poc_request("init", body.model_dump())
    
    # Start validating
    result = await engine_client.poc_request("start_validate", {})
    
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.post("/phase/generate")
async def start_generate(request: Request) -> dict:
    """Switch to generate mode."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Check if initialized
    status = await engine_client.poc_request("status", {})
    if status.get("state") == PoCState.IDLE.value:
        raise HTTPException(status_code=400, detail="PoC not initialized")
    
    result = await engine_client.poc_request("start_generate", {})
    
    # Start background generation loop
    task = asyncio.create_task(_generation_loop(engine_client))
    
    if not hasattr(request.app.state, '_poc_generation_task'):
        request.app.state._poc_generation_task = None
    if request.app.state._poc_generation_task:
        request.app.state._poc_generation_task.cancel()
    request.app.state._poc_generation_task = task
    
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.post("/phase/validate")
async def start_validate(request: Request) -> dict:
    """Switch to validate mode."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Check if initialized
    status = await engine_client.poc_request("status", {})
    if status.get("state") == PoCState.IDLE.value:
        raise HTTPException(status_code=400, detail="PoC not initialized")
    
    # Cancel generation task if running
    if hasattr(request.app.state, '_poc_generation_task'):
        task = request.app.state._poc_generation_task
        if task and not task.done():
            task.cancel()
        request.app.state._poc_generation_task = None
    
    result = await engine_client.poc_request("start_validate", {})
    
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


async def _generation_loop(engine_client):
    """Background task that runs batches until stopped."""
    try:
        while True:
            status = await engine_client.poc_request("status", {})
            if status.get("state") != PoCState.GENERATING.value:
                break
            
            await engine_client.poc_request("run_batch", {})
            await asyncio.sleep(0)  # Yield to event loop
    except asyncio.CancelledError:
        pass  # Normal cancellation on stop/reinit


@router.post("/stop")
async def stop_round(request: Request) -> dict:
    """Stop current PoC round."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Cancel generation task if running
    if hasattr(request.app.state, '_poc_generation_task'):
        task = request.app.state._poc_generation_task
        if task and not task.done():
            task.cancel()
        request.app.state._poc_generation_task = None
    
    result = await engine_client.poc_request("stop", {})
    
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.get("/status", response_model=PoCStatusResponse)
async def get_status(request: Request) -> PoCStatusResponse:
    """Get current PoC status."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    status = await engine_client.poc_request("status", {})
    return PoCStatusResponse(**status)


@router.post("/validate", response_model=PoCValidateResponse)
async def validate_nonces(request: Request, body: PoCValidateRequest) -> PoCValidateResponse:
    """Validate submitted nonces by recomputing distances.
    
    Accepts full ProofBatch format (matching original API).
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Check that we have a round configured
    status = await engine_client.poc_request("status", {})
    if status.get("state") == PoCState.IDLE.value:
        raise HTTPException(status_code=400, detail="No round configured")
    
    result = await engine_client.poc_request("validate", {
        "nonces": body.nonces,
        "public_key": body.public_key,
    })
    
    return PoCValidateResponse(
        nonces=result["nonces"],
        distances=result["distances"],
        valid=result["valid"],
    )


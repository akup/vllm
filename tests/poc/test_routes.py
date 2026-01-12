"""Tests for PoC API routes."""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.poc.routes import router, _poc_tasks, _is_generation_active, _get_next_nonces
from vllm.poc.generate_queue import GenerateJob, GenerateResult, get_queue, clear_queue, POC_MAX_QUEUED_NONCES
from vllm.poc.config import PoCState


async def _mock_generation_loop(engine_client, stop_event, callback_sender, config, stats):
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass


@pytest.fixture
def mock_engine_client():
    client = AsyncMock()
    client.poc_request.return_value = {"artifacts": []}
    return client


@pytest.fixture
def app_with_poc(mock_engine_client):
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = mock_engine_client
    app.state.poc_enabled = True
    app.state.poc_deployed = {"model": "test-model", "seq_len": 256, "k_dim": 12}
    mock_base_path = MagicMock()
    mock_base_path.model_path = "test-model"
    mock_base_path.name = "test-model"
    mock_serving_models = MagicMock()
    mock_serving_models.base_model_paths = [mock_base_path]
    app.state.openai_serving_models = mock_serving_models
    return app


@pytest.fixture
def client(app_with_poc):
    _poc_tasks.clear()
    with patch('vllm.poc.routes._generation_loop', _mock_generation_loop):
        yield TestClient(app_with_poc)
    for app_id, tasks in list(_poc_tasks.items()):
        if tasks.get("stop_event"):
            tasks["stop_event"].set()
        if tasks.get("gen_task"):
            tasks["gen_task"].cancel()
    _poc_tasks.clear()


class TestPoCInitGenerate:
    def test_init_generate_starts_generation(self, client, mock_engine_client):
        mock_engine_client.poc_request.return_value = {"artifacts": [{"nonce": 0, "vector_b64": "AAAA"}]}
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "batch_size": 32,
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12},
        })
        assert response.status_code == 200
        assert response.json()["status"] == "OK"
        assert response.json()["pow_status"]["status"] == "GENERATING"

    def test_init_generate_conflict_when_already_generating(self, client, app_with_poc):
        app_id = id(app_with_poc)
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _poc_tasks[app_id] = {"gen_task": mock_task, "stop_event": asyncio.Event(), "config": {}, "stats": {}}
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc456", "block_height": 101, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1,
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12},
        })
        assert response.status_code == 409

    def test_init_generate_params_mismatch(self, client):
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1,
            "params": {"model": "wrong-model", "seq_len": 256, "k_dim": 12},
        })
        assert response.status_code == 409

    def test_init_generate_extra_params_rejected(self, client):
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1,
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12, "extra": "bad"},
        })
        assert response.status_code == 422


class TestPoCGenerate:
    def test_generate_returns_artifacts(self, client, mock_engine_client):
        mock_engine_client.poc_request.return_value = {
            "artifacts": [{"nonce": 0, "vector_b64": "AAAA"}, {"nonce": 1, "vector_b64": "BBBB"}],
        }
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0, 1],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12}, "wait": True,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert len(response.json()["artifacts"]) == 2

    def test_generate_wait_false_returns_queued(self, client):
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0, 1, 2],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12}, "wait": False,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert response.json()["queued_count"] == 3

    def test_generate_with_validation_detects_mismatch(self, client, mock_engine_client):
        mock_engine_client.poc_request.return_value = {
            "artifacts": [{"nonce": 0, "vector_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}],
        }
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12}, "wait": True,
            "validation": {"artifacts": [{"nonce": 0, "vector_b64": "ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8"}]},
        })
        assert response.status_code == 200
        assert response.json()["n_mismatch"] == 1
        assert response.json()["fraud_detected"] is True

    def test_generate_validation_nonce_mismatch_error(self, client):
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0, 1],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12}, "wait": True,
            "validation": {"artifacts": [{"nonce": 0, "vector_b64": "AAA="}, {"nonce": 5, "vector_b64": "BBB="}]},
        })
        assert response.status_code == 400


class TestPoCStatus:
    def test_get_status_idle(self, client):
        response = client.get("/api/v1/pow/status")
        assert response.status_code == 200
        assert response.json()["status"] == "IDLE"

    def test_get_status_generating(self, client, app_with_poc):
        app_id = id(app_with_poc)
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _poc_tasks[app_id] = {
            "gen_task": mock_task, "stop_event": asyncio.Event(),
            "config": {"block_hash": "abc123", "block_height": 100, "public_key": "pk",
                       "node_id": 0, "node_count": 1, "seq_len": 256, "k_dim": 12},
            "stats": {"start_time": time.time(), "total_processed": 500},
        }
        response = client.get("/api/v1/pow/status")
        assert response.status_code == 200
        assert response.json()["status"] == "GENERATING"


class TestPoCStop:
    def test_stop_round(self, client, mock_engine_client):
        mock_engine_client.poc_request.return_value = {"artifacts": []}
        client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1,
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12},
        })
        response = client.post("/api/v1/pow/stop")
        assert response.status_code == 200
        assert response.json()["pow_status"]["status"] == "STOPPED"
        assert client.get("/api/v1/pow/status").json()["status"] == "IDLE"


class TestPoCDisabled:
    def test_routes_disabled_without_flag(self):
        app = FastAPI()
        app.include_router(router)
        app.state.poc_enabled = False
        app.state.engine_client = AsyncMock()
        test_client = TestClient(app)
        assert test_client.get("/api/v1/pow/status").status_code == 503

    def test_routes_require_engine(self):
        app = FastAPI()
        app.include_router(router)
        app.state.poc_enabled = True
        test_client = TestClient(app)
        response = test_client.post("/api/v1/pow/generate", json={
            "block_hash": "abc", "block_height": 100, "public_key": "pk",
            "node_id": 0, "node_count": 1, "nonces": [0],
            "params": {"model": "test", "seq_len": 256, "k_dim": 12}, "wait": True,
        })
        assert response.status_code == 503


class TestGenerationLoop:
    def test_nonce_generation_api_side(self):
        nonces, counter = _get_next_nonces(0, 4, 3)
        assert nonces == [0, 3, 6, 9]
        assert counter == 12


class TestGenerateQueue:
    def test_poll_unknown_request_returns_404(self, client):
        assert client.get("/api/v1/pow/generate/unknown-id").status_code == 404

    def test_poll_queued_request_returns_status(self, client):
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123", "block_height": 100, "public_key": "pubkey123",
            "node_id": 0, "node_count": 1, "nonces": [0],
            "params": {"model": "test-model", "seq_len": 256, "k_dim": 12}, "wait": False,
        })
        request_id = response.json()["request_id"]
        poll = client.get(f"/api/v1/pow/generate/{request_id}")
        assert poll.status_code == 200
        assert poll.json()["status"] in ["queued", "running", "completed"]


class TestQueueCap:
    @pytest.mark.asyncio
    async def test_queue_nonce_cap_enforced(self):
        queue = get_queue()
        await queue.clear_all()
        mock_client = AsyncMock()
        big_job = GenerateJob(
            request_id="big", engine_client=mock_client, app_id=1,
            block_hash="abc", block_height=100, public_key="pk",
            node_id=0, node_count=1, nonces=list(range(POC_MAX_QUEUED_NONCES + 1)),
            seq_len=256, k_dim=12, batch_size=1000,
        )
        assert await queue.enqueue(big_job) is None
        await queue.clear_all()


class TestGenerateQueueIntegration:
    @pytest.mark.asyncio
    async def test_queue_process_job(self):
        from vllm.poc.generate_queue import GenerateQueue
        queue = GenerateQueue()
        mock_client = AsyncMock()
        mock_client.poc_request.return_value = {"artifacts": [{"nonce": 0, "vector_b64": "AAAA"}]}
        job = GenerateJob(
            request_id="job1", engine_client=mock_client, app_id=1,
            block_hash="abc", block_height=100, public_key="pk",
            node_id=0, node_count=1, nonces=[0], seq_len=256, k_dim=12, batch_size=10,
        )
        result = await queue._process_job(job)
        assert result["status"] == "completed"

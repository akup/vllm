"""Tests for PoC API routes (artifact-based protocol)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.poc.routes import router
from vllm.poc.config import PoCState


# Mock class that simulates MQLLMEngineClient for tests
class _MockMQLLMEngineClient:
    """Mock that passes the MP engine requirement check."""
    pass


@pytest.fixture
def mock_engine_client():
    """Create a mock engine client for testing.
    
    Creates a mock that will pass the isinstance check for MP engine.
    """
    client = AsyncMock()
    
    # Default status response
    client.poc_request.return_value = {
        "status": PoCState.IDLE.value,
        "config": None,
        "stats": None,
    }
    
    return client


@pytest.fixture
def app_with_poc(mock_engine_client):
    """Create a FastAPI app with PoC router and mocked engine."""
    app = FastAPI()
    app.include_router(router)
    
    # Set up app state
    app.state.engine_client = mock_engine_client
    app.state.poc_enabled = True
    app.state.poc_deployed = {
        "model": "test-model",
        "seq_len": 256,
        "k_dim": 12,
    }
    
    # Set up mock openai_serving_models for params checking
    mock_base_path = MagicMock()
    mock_base_path.model_path = "test-model"
    mock_base_path.name = "test-model"
    
    mock_serving_models = MagicMock()
    mock_serving_models.base_model_paths = [mock_base_path]
    app.state.openai_serving_models = mock_serving_models
    
    return app


@pytest.fixture
def client(app_with_poc, mock_engine_client):
    """Create test client that patches the MP engine check."""
    # Patch the check to accept our mock client by making
    # MQLLMEngineClient point to AsyncMock's type
    with patch('vllm.engine.multiprocessing.client.MQLLMEngineClient', type(mock_engine_client)):
        yield TestClient(app_with_poc)


class TestPoCInitGenerate:
    def test_init_generate_starts_generation(self, client, mock_engine_client):
        """Test /init/generate starts background generation."""
        mock_engine_client.poc_request.side_effect = [
            # First call: status check
            {"status": PoCState.IDLE.value, "config": None},
            # Second call: init
            {"status": "OK", "pow_status": {"status": "IDLE"}},
            # Third call: start_generate
            {"status": "OK", "pow_status": {"status": "GENERATING"}},
        ]
        
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "batch_size": 32,
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"

    def test_init_generate_conflict_when_already_generating(self, client, mock_engine_client):
        """Test /init/generate returns 409 when already generating."""
        mock_engine_client.poc_request.return_value = {
            "status": PoCState.GENERATING.value,
        }
        
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
        })
        
        assert response.status_code == 409
        assert "Already generating" in response.json()["detail"]

    def test_init_generate_params_mismatch(self, client, mock_engine_client):
        """Test /init/generate returns 409 when params don't match deployed."""
        mock_engine_client.poc_request.return_value = {
            "status": PoCState.IDLE.value,
        }
        
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "params": {
                "model": "wrong-model",  # Doesn't match deployed
                "seq_len": 256,
                "k_dim": 12,
            },
        })
        
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "model mismatch" in detail.lower()
        assert "wrong-model" in detail


class TestPoCGenerate:
    def test_generate_returns_artifacts(self, client, mock_engine_client):
        """Test /generate computes artifacts for given nonces."""
        mock_engine_client.poc_request.side_effect = [
            # First call: status check
            {"status": PoCState.IDLE.value},
            # Second call: generate_artifacts
            {
                "artifacts": [
                    {"nonce": 0, "vector_b64": "AAAAAAA="},
                    {"nonce": 1, "vector_b64": "BBBBBBB="},
                ],
            },
        ]
        
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1],
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": True,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert len(data["artifacts"]) == 2
        assert data["encoding"]["dtype"] == "f16"

    def test_generate_conflict_when_busy(self, client, mock_engine_client):
        """Test /generate returns 409 when /init/generate is running."""
        mock_engine_client.poc_request.return_value = {
            "status": PoCState.GENERATING.value,
        }
        
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1],
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": True,
        })
        
        assert response.status_code == 409
        assert "Busy" in response.json()["detail"]

    def test_generate_wait_false_returns_queued(self, client, mock_engine_client):
        """Test /generate with wait=false returns queued status."""
        mock_engine_client.poc_request.return_value = {
            "status": PoCState.IDLE.value,
        }
        
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1, 2],
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": False,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["queued_count"] == 3

    def test_generate_with_validation_detects_mismatch(self, client, mock_engine_client):
        """Test /generate with validation field performs comparison."""
        # 12-dim vectors: all zeros vs all ones (very different)
        # zeros: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        # ones:  ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8
        mock_engine_client.poc_request.side_effect = [
            # First call: status check
            {"status": PoCState.IDLE.value},
            # Second call: generate_artifacts - returns zeros
            {
                "artifacts": [
                    {"nonce": 0, "vector_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
                    {"nonce": 1, "vector_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
                ],
            },
        ]
        
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1],
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": True,
            "validation": {
                "artifacts": [
                    {"nonce": 0, "vector_b64": "ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8"},  # ones - different
                    {"nonce": 1, "vector_b64": "ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8"},
                ],
            },
            "stat_test": {
                "dist_threshold": 0.02,
                "p_mismatch": 0.001,
                "fraud_threshold": 0.01,
            },
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["n_mismatch"] == 2
        assert data["fraud_detected"] is True

    def test_generate_validation_nonce_mismatch_error(self, client, mock_engine_client):
        """Test /generate returns 400 if validation nonces don't match."""
        mock_engine_client.poc_request.return_value = {
            "status": PoCState.IDLE.value,
        }
        
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1],  # Two nonces
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": True,
            "validation": {
                "artifacts": [
                    {"nonce": 0, "vector_b64": "AAA="},
                    {"nonce": 5, "vector_b64": "BBB="},  # Wrong nonce
                ],
            },
        })
        
        assert response.status_code == 400
        assert "must match" in response.json()["detail"]


class TestPoCStatus:
    def test_get_status(self, client, mock_engine_client):
        """Test getting PoC status."""
        mock_engine_client.poc_request.return_value = {
            "status": "GENERATING",
            "config": {
                "block_hash": "abc123",
                "block_height": 100,
                "k_dim": 12,
            },
            "stats": {
                "total_processed": 500,
                "nonces_per_second": 20.0,
            },
        }
        
        response = client.get("/api/v1/pow/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "GENERATING"
        assert data["stats"]["total_processed"] == 500


class TestPoCStop:
    def test_stop_round(self, client, mock_engine_client):
        """Test stopping a PoC round."""
        mock_engine_client.poc_request.return_value = {
            "status": "OK",
            "pow_status": {"status": "STOPPED"},
        }
        
        response = client.post("/api/v1/pow/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"


class TestPoCDisabled:
    def test_routes_disabled_without_flag(self):
        """Test that routes return 503 when PoC is not enabled."""
        app = FastAPI()
        app.include_router(router)
        
        app.state.poc_enabled = False
        app.state.engine_client = AsyncMock()
        
        client = TestClient(app)
        
        response = client.get("/api/v1/pow/status")
        assert response.status_code == 503
        assert "not enabled" in response.json()["detail"]

    def test_routes_require_engine(self):
        """Test that routes return 503 when engine is not available."""
        app = FastAPI()
        app.include_router(router)
        
        app.state.poc_enabled = True
        # No engine_client
        
        client = TestClient(app)
        
        response = client.get("/api/v1/pow/status")
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]


class TestRemovedEndpoints:
    """Test that old endpoints are removed (404)."""
    
    def test_old_init_removed(self, client):
        """Test /init endpoint no longer exists."""
        response = client.post("/api/v1/pow/init", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "r_target": 0.5,
        })
        assert response.status_code == 404

    def test_old_init_validate_removed(self, client):
        """Test /init/validate endpoint no longer exists."""
        response = client.post("/api/v1/pow/init/validate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "r_target": 0.5,
        })
        assert response.status_code == 404

    def test_old_phase_generate_removed(self, client):
        """Test /phase/generate endpoint no longer exists."""
        response = client.post("/api/v1/pow/phase/generate")
        assert response.status_code == 404

    def test_old_phase_validate_removed(self, client):
        """Test /phase/validate endpoint no longer exists."""
        response = client.post("/api/v1/pow/phase/validate")
        assert response.status_code == 404

    def test_old_validate_removed(self, client):
        """Test /validate endpoint no longer exists."""
        response = client.post("/api/v1/pow/validate", json={
            "public_key": "pubkey123",
            "block_hash": "abc123",
            "block_height": 100,
            "nonces": [1, 2, 3],
            "dist": [0.1, 0.2, 0.3],
            "node_id": 0,
        })
        assert response.status_code == 404

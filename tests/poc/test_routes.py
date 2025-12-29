"""Tests for PoC API routes."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.poc.routes import (
    router,
    PoCInitRequest,
    PoCValidateRequest,
    PoCStatusResponse,
)

# Note: PoCValidateResponse was removed as /validate is now fire-and-forget
from vllm.poc.config import PoCState


@pytest.fixture
def mock_engine_client():
    """Create a mock engine client for testing."""
    client = AsyncMock()
    
    # Default status response
    client.poc_request.return_value = {
        "state": PoCState.IDLE.value,
        "valid_nonces": [],
        "valid_distances": [],
        "total_checked": 0,
        "total_valid": 0,
        "elapsed_seconds": 0.0,
        "rate_per_second": 0.0,
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
    
    return app


@pytest.fixture
def client(app_with_poc):
    """Create test client."""
    return TestClient(app_with_poc)


class TestPoCInit:
    def test_init_round(self, client, mock_engine_client):
        """Test initializing a PoC round."""
        mock_engine_client.poc_request.return_value = {
            "status": "initialized",
            "pow_status": {
                "state": PoCState.IDLE.value,
                "valid_nonces": [],
                "valid_distances": [],
                "total_checked": 0,
                "total_valid": 0,
                "elapsed_seconds": 0.0,
                "rate_per_second": 0.0,
            }
        }
        
        response = client.post("/api/v1/pow/init", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "r_target": 0.5,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert "pow_status" in data
        
        # Verify engine was called correctly
        mock_engine_client.poc_request.assert_called_once()
        args = mock_engine_client.poc_request.call_args
        assert args[0][0] == "init"

    def test_init_generate(self, client, mock_engine_client):
        """Test init with generate phase."""
        mock_engine_client.poc_request.return_value = {
            "status": "generating",
            "pow_status": {
                "state": PoCState.GENERATING.value,
                "valid_nonces": [],
                "valid_distances": [],
                "total_checked": 0,
                "total_valid": 0,
                "elapsed_seconds": 0.0,
                "rate_per_second": 0.0,
            }
        }
        
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "r_target": 0.5,
            "node_id": 0,
            "node_count": 1,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"

    def test_init_generate_requires_node_info(self, client, mock_engine_client):
        """Test that init/generate requires node_id and node_count."""
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "r_target": 0.5,
            # Missing node_id and node_count (default to -1)
        })
        
        assert response.status_code == 400
        assert "Node ID" in response.json()["detail"]

    def test_init_validate(self, client, mock_engine_client):
        """Test init with validate phase."""
        mock_engine_client.poc_request.return_value = {
            "status": "validating",
            "pow_status": {
                "state": PoCState.VALIDATING.value,
                "valid_nonces": [],
                "valid_distances": [],
                "total_checked": 0,
                "total_valid": 0,
                "elapsed_seconds": 0.0,
                "rate_per_second": 0.0,
            }
        }
        
        response = client.post("/api/v1/pow/init/validate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "r_target": 0.5,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"


class TestPoCPhase:
    def test_switch_to_generate(self, client, mock_engine_client):
        """Test switching to generate phase."""
        mock_engine_client.poc_request.side_effect = [
            # First call: status check (needs to be initialized, must include r_target)
            {"state": PoCState.VALIDATING.value, "r_target": 1.5},
            # Second call: start_generate
            {"status": "generating", "pow_status": {"state": PoCState.GENERATING.value}},
        ]
        
        response = client.post("/api/v1/pow/phase/generate")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"

    def test_switch_to_validate(self, client, mock_engine_client):
        """Test switching to validate phase."""
        mock_engine_client.poc_request.side_effect = [
            # First call: status check
            {"state": PoCState.GENERATING.value},
            # Second call: start_validate
            {"status": "validating", "pow_status": {"state": PoCState.VALIDATING.value}},
        ]
        
        response = client.post("/api/v1/pow/phase/validate")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"

    def test_phase_requires_init(self, client, mock_engine_client):
        """Test that phase switch requires initialization."""
        # Status returns IDLE state
        mock_engine_client.poc_request.return_value = {
            "state": PoCState.IDLE.value
        }
        
        response = client.post("/api/v1/pow/phase/generate")
        
        assert response.status_code == 400
        assert "not initialized" in response.json()["detail"]


class TestPoCStop:
    def test_stop_round(self, client, mock_engine_client):
        """Test stopping a PoC round."""
        mock_engine_client.poc_request.return_value = {
            "status": "stopped",
            "pow_status": {
                "state": PoCState.STOPPED.value,
                "valid_nonces": [1, 2, 3],
                "valid_distances": [0.1, 0.2, 0.3],
                "total_checked": 100,
                "total_valid": 3,
                "elapsed_seconds": 10.5,
                "rate_per_second": 9.5,
            }
        }
        
        response = client.post("/api/v1/pow/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert "pow_status" in data


class TestPoCStatus:
    def test_get_status(self, client, mock_engine_client):
        """Test getting PoC status."""
        mock_engine_client.poc_request.return_value = {
            "state": PoCState.GENERATING.value,
            "valid_nonces": [1, 5, 10],
            "valid_distances": [0.1, 0.2, 0.3],
            "total_checked": 500,
            "total_valid": 3,
            "elapsed_seconds": 25.0,
            "rate_per_second": 20.0,
        }
        
        response = client.get("/api/v1/pow/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "GENERATING"
        assert data["valid_nonces"] == [1, 5, 10]
        assert data["total_checked"] == 500
        assert data["total_valid"] == 3


class TestPoCValidate:
    def test_validate_nonces_fire_and_forget(self, client, mock_engine_client):
        """Test validating nonces (fire-and-forget, matching original API)."""
        # Mock status to show round is configured
        def mock_poc_request(action, payload):
            if action == "status":
                return {"state": PoCState.VALIDATING.value}
            elif action == "queue_validation":
                return {"status": "queued"}
            return {}
        
        mock_engine_client.poc_request.side_effect = mock_poc_request
        
        response = client.post("/api/v1/pow/validate", json={
            "public_key": "pubkey123",
            "block_hash": "abc123",
            "block_height": 100,
            "nonces": [1, 2, 3],
            "dist": [0.1, 0.6, 0.2],
            "node_id": 0,
        })
        
        assert response.status_code == 200
        data = response.json()
        # Fire-and-forget returns just status OK
        assert data["status"] == "OK"

    def test_validate_requires_init(self, client, mock_engine_client):
        """Test that validate requires round to be configured."""
        mock_engine_client.poc_request.return_value = {
            "state": PoCState.IDLE.value
        }
        
        response = client.post("/api/v1/pow/validate", json={
            "public_key": "pubkey123",
            "block_hash": "abc123",
            "block_height": 100,
            "nonces": [1, 2, 3],
            "dist": [0.1, 0.6, 0.2],
            "node_id": 0,
        })
        
        assert response.status_code == 400
        assert "No round configured" in response.json()["detail"]


class TestPoCDisabled:
    def test_routes_disabled_without_flag(self):
        """Test that routes return 503 when PoC is not enabled."""
        app = FastAPI()
        app.include_router(router)
        
        # PoC not enabled
        app.state.poc_enabled = False
        app.state.engine_client = AsyncMock()
        
        client = TestClient(app)
        
        # All endpoints should return 503
        response = client.post("/api/v1/pow/init", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "r_target": 0.5,
        })
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


class TestPoCRequestModels:
    def test_init_request_defaults(self):
        """Test PoCInitRequest default values."""
        req = PoCInitRequest(
            block_hash="abc",
            block_height=100,
            public_key="pub",
            r_target=0.5,
        )
        
        assert req.fraud_threshold == 0.01
        assert req.node_id == -1
        assert req.node_count == -1
        assert req.batch_size == 32
        assert req.seq_len == 256
        assert req.callback_url is None

    def test_validate_request(self):
        """Test PoCValidateRequest model."""
        req = PoCValidateRequest(
            public_key="pub",
            block_hash="abc",
            block_height=100,
            nonces=[1, 2, 3],
            dist=[0.1, 0.2, 0.3],
            node_id=0,
        )
        
        assert req.nonces == [1, 2, 3]
        assert req.dist == [0.1, 0.2, 0.3]

    def test_status_response(self):
        """Test PoCStatusResponse model."""
        resp = PoCStatusResponse(
            state="generating",
            valid_nonces=[1, 2],
            valid_distances=[0.1, 0.2],
            total_checked=100,
            total_valid=2,
            elapsed_seconds=10.0,
            rate_per_second=10.0,
        )
        
        assert resp.state == "generating"
        assert resp.total_checked == 100


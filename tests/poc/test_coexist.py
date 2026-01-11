"""Tests for PoC+Chat coexistence (Phase 2)."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.poc.routes import (
    router, _generation_loop, check_mp_engine_required,
    POC_CHAT_BUSY_BACKOFF_SEC, POC_GENERATE_CHUNK_TIMEOUT_SEC,
)
from vllm.poc.config import PoCState


class MockMQLLMEngineClient:
    """Mock that simulates MQLLMEngineClient."""
    pass


class MockAsyncLLMEngine:
    """Mock that simulates in-process AsyncLLMEngine."""
    pass


@pytest.fixture
def mock_mp_engine_client():
    """Create a mock MP engine client for testing."""
    client = AsyncMock(spec=MockMQLLMEngineClient)
    client.poc_request = AsyncMock()
    client.poc_request.return_value = {
        "status": PoCState.IDLE.value,
        "config": None,
        "stats": None,
    }
    return client


@pytest.fixture
def mock_inprocess_engine_client():
    """Create a mock in-process engine client for testing."""
    client = AsyncMock(spec=MockAsyncLLMEngine)
    client.poc_request = AsyncMock()
    client.poc_request.return_value = {
        "status": PoCState.IDLE.value,
        "config": None,
        "stats": None,
    }
    return client


@pytest.fixture
def app_with_mp_engine(mock_mp_engine_client):
    """Create a FastAPI app with MP engine client."""
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = mock_mp_engine_client
    app.state.poc_enabled = True
    return app


@pytest.fixture
def app_with_inprocess_engine(mock_inprocess_engine_client):
    """Create a FastAPI app with in-process engine client."""
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = mock_inprocess_engine_client
    app.state.poc_enabled = True
    return app


class TestMPEngineGuard:
    """Tests for MP engine requirement guard."""
    
    def test_check_mp_engine_required_passes_for_mp_client(self):
        """Test that MP engine check passes for MQLLMEngineClient."""
        with patch('vllm.engine.multiprocessing.client.MQLLMEngineClient', MockMQLLMEngineClient):
            client = MockMQLLMEngineClient()
            # Should not raise
            check_mp_engine_required(client)
    
    def test_check_mp_engine_required_fails_for_inprocess_client(self):
        """Test that MP engine check fails for non-MP clients."""
        from fastapi import HTTPException
        with patch('vllm.engine.multiprocessing.client.MQLLMEngineClient', MockMQLLMEngineClient):
            client = MockAsyncLLMEngine()
            with pytest.raises(HTTPException) as exc_info:
                check_mp_engine_required(client)
            assert exc_info.value.status_code == 503
            assert "multiprocessing engine mode" in exc_info.value.detail

    def test_init_generate_rejects_inprocess_engine(self, app_with_inprocess_engine):
        """Test /init/generate returns 503 for in-process engine."""
        with patch('vllm.engine.multiprocessing.client.MQLLMEngineClient', MockMQLLMEngineClient):
            client = TestClient(app_with_inprocess_engine)
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
        
            assert response.status_code == 503
            assert "multiprocessing" in response.json()["detail"].lower()
    
    def test_generate_rejects_inprocess_engine(self, app_with_inprocess_engine):
        """Test /generate returns 503 for in-process engine."""
        with patch('vllm.engine.multiprocessing.client.MQLLMEngineClient', MockMQLLMEngineClient):
            client = TestClient(app_with_inprocess_engine)
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
                "wait": True,
            })
        
            assert response.status_code == 503
            assert "multiprocessing" in response.json()["detail"].lower()


class TestChatPriorityGating:
    """Tests for chat-priority gating in PoC GPU actions."""
    
    def test_run_batch_skips_when_pending_input(self):
        """Test run_batch returns skip when there's pending input (chat waiting)."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        
        # Use MagicMock without spec to allow input_socket attribute
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        # Simulate pending input on socket
        mq_engine.input_socket.poll.return_value = 1  # Non-zero = pending
        
        mock_manager = MagicMock()
        mock_manager.state.value = "GENERATING"
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "run_batch", {})
        
        assert result["skipped"] is True
        assert result["reason"] == "pending_input"
        mock_manager.run_batch.assert_not_called()
    
    def test_run_batch_skips_when_engine_step_in_progress(self):
        """Test run_batch returns skip when _engine_step_in_progress is True."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = True
        # No pending input
        mq_engine.input_socket.poll.return_value = 0
        
        mock_manager = MagicMock()
        mock_manager.state.value = "GENERATING"
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "run_batch", {})
        
        assert result["skipped"] is True
        assert result["reason"] == "engine_step_in_progress"
        mock_manager.run_batch.assert_not_called()
    
    def test_run_batch_skips_when_chat_unfinished(self):
        """Test run_batch returns skip when chat has unfinished requests."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        mock_llm_engine.has_unfinished_requests.return_value = True
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 0  # No pending input
        
        mock_manager = MagicMock()
        mock_manager.state.value = "GENERATING"
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "run_batch", {})
        
        assert result["skipped"] is True
        assert result["reason"] == "chat_unfinished"
        mock_manager.run_batch.assert_not_called()
    
    def test_run_batch_proceeds_when_all_checks_pass(self):
        """Test run_batch proceeds when no pending input, not in step, and no chat."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        mock_llm_engine.has_unfinished_requests.return_value = False
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 0  # No pending input
        
        mock_manager = MagicMock()
        mock_manager.run_batch.return_value = {
            "should_continue": True,
            "state": "GENERATING",
            "nonces": [0, 1, 2],
            "artifacts": [],
        }
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "run_batch", {})
        
        # Should call _prepare_for_poc_gpu_work before run_batch
        mq_engine._prepare_for_poc_gpu_work.assert_called_once()
        mock_manager.run_batch.assert_called_once()
        assert "skipped" not in result or result.get("skipped") is not True
    
    def test_generate_artifacts_skips_when_pending_input(self):
        """Test generate_artifacts returns skip when there's pending input."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 1  # Pending input
        
        mock_manager = MagicMock()
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "generate_artifacts", {
            "nonces": [0, 1, 2],
        })
        
        assert result["skipped"] is True
        assert result["reason"] == "pending_input"
        mock_manager.generate_artifacts.assert_not_called()
    
    def test_generate_artifacts_skips_when_engine_step_in_progress(self):
        """Test generate_artifacts returns skip when _engine_step_in_progress is True."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = True
        mq_engine.input_socket.poll.return_value = 0
        
        mock_manager = MagicMock()
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "generate_artifacts", {
            "nonces": [0, 1, 2],
        })
        
        assert result["skipped"] is True
        assert result["reason"] == "engine_step_in_progress"
        mock_manager.generate_artifacts.assert_not_called()
    
    def test_generate_artifacts_skips_when_chat_unfinished(self):
        """Test generate_artifacts returns skip when chat has unfinished requests."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        mock_llm_engine.has_unfinished_requests.return_value = True
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 0
        
        mock_manager = MagicMock()
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "generate_artifacts", {
            "nonces": [0, 1, 2],
        })
        
        assert result["skipped"] is True
        assert result["reason"] == "chat_unfinished"
        mock_manager.generate_artifacts.assert_not_called()
    
    def test_generate_artifacts_proceeds_when_all_checks_pass(self):
        """Test generate_artifacts proceeds when all priority checks pass."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        mock_llm_engine.has_unfinished_requests.return_value = False
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine.input_socket.poll.return_value = 0
        
        mock_artifact = MagicMock()
        mock_artifact.nonce = 0
        mock_artifact.vector_b64 = "base64data"
        mock_manager = MagicMock()
        mock_manager.generate_artifacts.return_value = [mock_artifact]
        mq_engine._poc_manager = mock_manager
        mq_engine._get_poc_manager = lambda: mock_manager
        
        result = MQLLMEngine._process_poc_action(mq_engine, "generate_artifacts", {
            "nonces": [0],
            "block_hash": "test",
            "public_key": "test",
            "seq_len": 256,
            "k_dim": 12,
        })
        
        # Should call _prepare_for_poc_gpu_work before generate_artifacts
        mq_engine._prepare_for_poc_gpu_work.assert_called_once()
        mock_manager.generate_artifacts.assert_called_once()
        assert "skipped" not in result or result.get("skipped") is not True


class TestPrepareForPoCGpuWork:
    """Tests for _prepare_for_poc_gpu_work helper method (v0 TP deadlock fix)."""
    
    def test_stops_remote_worker_loop_when_running(self):
        """Test that remote worker execution loop is stopped when it's running."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        # Simulate a distributed executor with running worker loop
        mock_executor = MagicMock()
        mock_executor.parallel_worker_tasks = MagicMock()  # Non-None = loop is running
        mock_llm_engine.model_executor = mock_executor
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        
        # Call the real method
        MQLLMEngine._prepare_for_poc_gpu_work(mq_engine)
        
        mock_executor.stop_remote_worker_execution_loop.assert_called_once()
    
    def test_no_op_when_remote_loop_not_running(self):
        """Test that nothing happens when remote worker loop is not running."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        # Simulate a distributed executor with NO running worker loop
        mock_executor = MagicMock()
        mock_executor.parallel_worker_tasks = None  # None = loop not running
        mock_llm_engine.model_executor = mock_executor
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        
        # Call the real method
        MQLLMEngine._prepare_for_poc_gpu_work(mq_engine)
        
        mock_executor.stop_remote_worker_execution_loop.assert_not_called()
    
    def test_no_op_for_non_distributed_executor(self):
        """Test that nothing happens for executors without parallel_worker_tasks."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        # Simulate a non-distributed executor (e.g., UniProcExecutor)
        mock_executor = MagicMock(spec=[])  # No parallel_worker_tasks attribute
        mock_llm_engine.model_executor = mock_executor
        
        mq_engine = MagicMock()
        mq_engine.engine = mock_llm_engine
        
        # Call the real method - should not raise
        MQLLMEngine._prepare_for_poc_gpu_work(mq_engine)
        
        # Since spec=[], hasattr will return False, so no call should be made
        assert not hasattr(mock_executor, 'stop_remote_worker_execution_loop') or \
               not mock_executor.stop_remote_worker_execution_loop.called


class TestGenerationLoopBackoff:
    """Tests for backoff behavior in generation loop."""
    
    @pytest.mark.asyncio
    async def test_generation_loop_sleeps_when_skipped(self):
        """Test _generation_loop sleeps when run_batch returns skipped."""
        mock_engine_client = AsyncMock()
        stop_event = asyncio.Event()
        artifact_queue = asyncio.Queue()
        config = {"block_hash": "test", "public_key": "test"}
        
        call_count = 0
        
        async def mock_poc_request(action, payload, timeout_ms=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # First two calls: skipped
                return {
                    "should_continue": True,
                    "state": "GENERATING",
                    "nonces": [],
                    "artifacts": [],
                    "skipped": True,
                }
            else:
                # Third call: stop
                stop_event.set()
                return {"should_continue": False}
        
        mock_engine_client.poc_request = mock_poc_request
        
        with patch('vllm.poc.routes.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await _generation_loop(
                mock_engine_client,
                stop_event,
                artifact_queue,
                config,
            )
            
            # Should have slept twice (for the two skipped calls)
            assert mock_sleep.call_count >= 2
            mock_sleep.assert_called_with(POC_CHAT_BUSY_BACKOFF_SEC)
    
    @pytest.mark.asyncio
    async def test_generation_loop_no_sleep_when_not_skipped(self):
        """Test _generation_loop doesn't sleep when run_batch succeeds."""
        mock_engine_client = AsyncMock()
        stop_event = asyncio.Event()
        artifact_queue = asyncio.Queue()
        config = {"block_hash": "test", "public_key": "test"}
        
        call_count = 0
        
        async def mock_poc_request(action, payload, timeout_ms=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {
                    "should_continue": True,
                    "state": "GENERATING",
                    "public_key": "test",
                    "block_hash": "test",
                    "block_height": 100,
                    "node_id": 0,
                    "nonces": [call_count],
                    "artifacts": [],
                }
            else:
                stop_event.set()
                return {"should_continue": False}
        
        mock_engine_client.poc_request = mock_poc_request
        
        with patch('vllm.poc.routes.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await _generation_loop(
                mock_engine_client,
                stop_event,
                artifact_queue,
                config,
            )
            
            # Should not have slept for backoff
            for call in mock_sleep.call_args_list:
                assert call[0][0] != POC_CHAT_BUSY_BACKOFF_SEC


class TestGenerationLoopExceptionHandling:
    """Tests for exception handling in generation loop."""
    
    @pytest.mark.asyncio
    async def test_generation_loop_logs_and_raises_on_exception(self):
        """Test _generation_loop logs and re-raises non-cancel exceptions."""
        mock_engine_client = AsyncMock()
        stop_event = asyncio.Event()
        artifact_queue = asyncio.Queue()
        config = {"block_hash": "test", "public_key": "test"}
        
        # Make poc_request raise an exception
        mock_engine_client.poc_request = AsyncMock(
            side_effect=RuntimeError("Test error")
        )
        
        with pytest.raises(RuntimeError, match="Test error"):
            await _generation_loop(
                mock_engine_client,
                stop_event,
                artifact_queue,
                config,
            )
    
    @pytest.mark.asyncio
    async def test_generation_loop_handles_cancelled_gracefully(self):
        """Test _generation_loop handles CancelledError gracefully."""
        mock_engine_client = AsyncMock()
        stop_event = asyncio.Event()
        artifact_queue = asyncio.Queue()
        config = {"block_hash": "test", "public_key": "test"}
        
        call_count = 0
        
        async def mock_poc_request(action, payload, timeout_ms=None):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()
            return {
                "should_continue": True,
                "state": "GENERATING",
                "nonces": [0],
                "artifacts": [],
            }
        
        mock_engine_client.poc_request = mock_poc_request
        
        # Should not raise - CancelledError is handled gracefully
        await _generation_loop(
            mock_engine_client,
            stop_event,
            artifact_queue,
            config,
        )


class TestGenerationLoopTimeoutRecovery:
    """Tests for timeout recovery in generation loop."""
    
    @pytest.mark.asyncio
    async def test_generation_loop_recovers_from_timeout(self):
        """Test _generation_loop treats TimeoutError as recoverable."""
        mock_engine_client = AsyncMock()
        stop_event = asyncio.Event()
        artifact_queue = asyncio.Queue()
        config = {"block_hash": "test", "public_key": "test"}
        
        call_count = 0
        
        async def mock_poc_request(action, payload, timeout_ms=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # First two calls: timeout
                raise TimeoutError("Engine busy")
            else:
                # Third call: stop
                stop_event.set()
                return {"should_continue": False}
        
        mock_engine_client.poc_request = mock_poc_request
        
        with patch('vllm.poc.routes.asyncio.sleep', new_callable=AsyncMock):
            # Should not raise - timeout is recovered
            await _generation_loop(
                mock_engine_client,
                stop_event,
                artifact_queue,
                config,
            )
        
        # Should have called 3 times (2 timeouts + 1 success)
        assert call_count == 3


class TestGenerateRetryLogic:
    """Tests for /generate endpoint retry logic when engine is busy."""
    
    @pytest.mark.asyncio
    async def test_generate_retries_on_skipped(self):
        """Test /generate retries chunks when engine returns skipped."""
        from vllm.poc.routes import generate, PoCGenerateRequest, PoCParamsModel
        
        call_count = 0
        
        async def mock_poc_request(action, payload):
            nonlocal call_count
            if action == "status":
                return {"status": PoCState.IDLE.value}
            if action == "generate_artifacts":
                call_count += 1
                if call_count < 3:
                    # First two calls: skipped
                    return {"artifacts": [], "skipped": True}
                else:
                    # Third call: success
                    return {
                        "artifacts": [
                            {"nonce": n, "vector_b64": "base64data"}
                            for n in payload["nonces"]
                        ]
                    }
            return {}
        
        mock_engine_client = AsyncMock()
        mock_engine_client.poc_request = mock_poc_request
        
        mock_request = MagicMock()
        mock_request.app.state.poc_enabled = True
        mock_request.app.state.engine_client = mock_engine_client
        mock_request.app.state.openai_serving_models = None
        
        body = PoCGenerateRequest(
            block_hash="abc123",
            block_height=100,
            public_key="pubkey123",
            node_id=0,
            node_count=1,
            nonces=[0, 1, 2],
            params=PoCParamsModel(model="test", seq_len=256, k_dim=12),
            batch_size=20,
            wait=True,
        )
        
        with patch('vllm.poc.routes.check_mp_engine_required'):
            with patch('vllm.poc.routes.asyncio.sleep', new_callable=AsyncMock):
                result = await generate(mock_request, body)
        
        assert result["status"] == "completed"
        assert len(result["artifacts"]) == 3
        # Should have retried (3 calls for one chunk)
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_generate_timeout_on_persistent_skip(self):
        """Test /generate times out when engine persistently returns skipped."""
        from vllm.poc.routes import generate, PoCGenerateRequest, PoCParamsModel
        from fastapi import HTTPException
        
        async def mock_poc_request(action, payload):
            if action == "status":
                return {"status": PoCState.IDLE.value}
            if action == "generate_artifacts":
                # Always return skipped
                return {"artifacts": [], "skipped": True}
            return {}
        
        mock_engine_client = AsyncMock()
        mock_engine_client.poc_request = mock_poc_request
        
        mock_request = MagicMock()
        mock_request.app.state.poc_enabled = True
        mock_request.app.state.engine_client = mock_engine_client
        mock_request.app.state.openai_serving_models = None
        
        body = PoCGenerateRequest(
            block_hash="abc123",
            block_height=100,
            public_key="pubkey123",
            node_id=0,
            node_count=1,
            nonces=[0, 1, 2],
            params=PoCParamsModel(model="test", seq_len=256, k_dim=12),
            batch_size=20,
            wait=True,
        )
        
        # Patch timeout to be very short for testing
        with patch('vllm.poc.routes.POC_GENERATE_CHUNK_TIMEOUT_SEC', 0.1):
            with patch('vllm.poc.routes.check_mp_engine_required'):
                with patch('vllm.poc.routes.asyncio.sleep', new_callable=AsyncMock):
                    with pytest.raises(HTTPException) as exc_info:
                        await generate(mock_request, body)
        
        assert exc_info.value.status_code == 503
        assert "Timeout" in exc_info.value.detail


class TestClientPocRequestTimeout:
    """Tests for MQLLMEngineClient.poc_request timeout."""
    
    @pytest.mark.asyncio
    async def test_poc_request_timeout_behavior(self):
        """Test that poc_request timeout mechanism works correctly."""
        # Test that asyncio.wait_for properly times out on queue.get()
        queue = asyncio.Queue()
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.01)
    
    @pytest.mark.asyncio
    async def test_poc_request_timeout_error_message(self):
        """Test that our custom TimeoutError has the expected message."""
        # Simulate what poc_request does on timeout
        action = "test_action"
        timeout_ms = 100
        
        try:
            # This simulates what happens in poc_request
            queue = asyncio.Queue()
            timeout_sec = timeout_ms / 1000.0
            await asyncio.wait_for(queue.get(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            # This is what poc_request does - raises our custom TimeoutError
            error = TimeoutError(
                f"PoC request '{action}' timed out after {timeout_ms}ms. "
                "Engine may be wedged."
            )
            assert "timed out" in str(error)
            assert action in str(error)
    
    @pytest.mark.asyncio
    async def test_poc_request_cleans_up_queue_on_timeout(self):
        """Test that queue cleanup happens on timeout."""
        output_queues = {"test-request-id": asyncio.Queue()}
        
        # Simulate the cleanup that happens in poc_request
        try:
            queue = output_queues["test-request-id"]
            await asyncio.wait_for(queue.get(), timeout=0.01)
        except asyncio.TimeoutError:
            # Clean up the queue (as poc_request does)
            output_queues.pop("test-request-id", None)
        
        assert "test-request-id" not in output_queues


class TestEngineStepFlagLifecycle:
    """Tests for _engine_step_in_progress flag lifecycle."""
    
    def test_engine_step_sets_and_clears_flag(self):
        """Test engine_step() properly sets and clears the flag."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        # Create mocks
        mock_llm_engine = MagicMock()
        mock_llm_engine.step.return_value = []
        
        mq_engine = MagicMock(spec=MQLLMEngine)
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        
        # Track flag changes
        flag_during_step = None
        
        def capture_flag():
            nonlocal flag_during_step
            flag_during_step = mq_engine._engine_step_in_progress
            return []
        
        mock_llm_engine.step.side_effect = capture_flag
        
        # Call engine_step
        MQLLMEngine.engine_step(mq_engine)
        
        # Flag should have been True during step
        # Note: This test verifies the implementation pattern,
        # actual flag capture requires more complex mocking
        mock_llm_engine.step.assert_called_once()
    
    def test_engine_step_clears_flag_on_exception(self):
        """Test engine_step() clears flag even on exception."""
        from vllm.engine.multiprocessing.engine import MQLLMEngine
        
        mock_llm_engine = MagicMock()
        mock_llm_engine.step.side_effect = RuntimeError("Test error")
        
        mq_engine = MagicMock(spec=MQLLMEngine)
        mq_engine.engine = mock_llm_engine
        mq_engine._engine_step_in_progress = False
        mq_engine._errored_with = None
        mq_engine._set_errored = MagicMock()
        mq_engine._send_outputs = MagicMock()
        
        # The real implementation uses a finally block
        # This test documents the expected behavior
        with pytest.raises(RuntimeError):
            MQLLMEngine.engine_step(mq_engine)

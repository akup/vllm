"""Tests for PoCManager (artifact-based protocol)."""
import pytest
import time
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from vllm.poc.config import PoCConfig, PoCState
from vllm.poc.manager import PoCManager, PoCStats
from vllm.poc.data import decode_vector


class MockModelConfig:
    """Mock vLLM model config"""
    
    def get_hidden_size(self):
        return 128
    
    def get_vocab_size(self):
        return 1000


def create_mock_model_executor():
    """Create a mock model executor with driver_worker."""
    executor = MagicMock()
    executor.driver_worker = MagicMock()
    executor.driver_worker.device = torch.device("cpu")
    executor.collective_rpc = MagicMock(return_value=[None])
    return executor


def create_mock_vllm_config():
    """Create a mock VllmConfig for testing."""
    vllm_config = MagicMock()
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.parallel_config.data_parallel_size = 1
    return vllm_config


class TestPoCStats:
    def test_initial_state(self):
        stats = PoCStats()
        assert stats.total_processed == 0
        assert stats.elapsed == 0.0
        assert stats.nonces_per_second == 0.0
    
    def test_elapsed_calculation(self):
        stats = PoCStats(total_processed=100, start_time=time.time() - 10)
        assert stats.elapsed >= 10
        assert stats.elapsed < 11
    
    def test_rate_calculation(self):
        stats = PoCStats(total_processed=100, start_time=time.time() - 10)
        assert stats.nonces_per_second >= 9
        assert stats.nonces_per_second <= 11


class TestPoCManagerInit:
    @pytest.fixture
    def mock_executor(self):
        return create_mock_model_executor()
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_initial_state(self, manager):
        assert manager.state == PoCState.IDLE
        assert manager.config is None


class TestPoCManagerStateTransitions:
    @pytest.fixture
    def mock_executor(self):
        return create_mock_model_executor()
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    @pytest.fixture
    def config(self):
        return PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
        )
    
    def test_init_round_sets_config(self, manager, config):
        manager.init_round(config)
        
        assert manager.state == PoCState.IDLE
        assert manager.config == config
    
    def test_start_generate_sets_state(self, manager, config):
        manager.init_round(config)
        
        manager.start_generate()
        assert manager.state == PoCState.GENERATING
    
    def test_init_round_resets_counters(self, manager, config):
        manager.stats.total_processed = 100
        
        manager.init_round(config)
        
        assert manager.stats.total_processed == 0
    
    def test_init_round_raises_if_already_generating(self, manager, config):
        manager.init_round(config)
        manager.start_generate()
        
        with pytest.raises(RuntimeError, match="Round already in progress"):
            manager.init_round(config)
    
    def test_start_generate_requires_init(self, manager):
        with pytest.raises(RuntimeError, match="Round not initialized"):
            manager.start_generate()
    
    def test_stop_round_from_generating(self, manager, config):
        manager.init_round(config)
        manager.start_generate()
        
        manager.stop_round()
        assert manager.state == PoCState.STOPPED


class TestPoCManagerNonceGeneration:
    """Cross-check: Nonce iteration pattern."""
    
    @pytest.fixture
    def mock_executor(self):
        return create_mock_model_executor()
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_single_node_nonces(self, manager):
        """Single node gets sequential nonces: 0, 1, 2, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            node_id=0,
            node_count=1,
            batch_size=4,
        )
        manager.init_round(config)
        manager.start_generate()
        
        nonces1 = manager.get_next_nonces()
        assert nonces1 == [0, 1, 2, 3]
        
        nonces2 = manager.get_next_nonces()
        assert nonces2 == [4, 5, 6, 7]
    
    def test_multi_node_nonces_node0(self, manager):
        """Node 0 of 3 gets: 0, 3, 6, 9, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            node_id=0,
            node_count=3,
            batch_size=4,
        )
        manager.init_round(config)
        manager.start_generate()
        
        nonces = manager.get_next_nonces()
        assert nonces == [0, 3, 6, 9]
    
    def test_multi_node_nonces_node1(self, manager):
        """Node 1 of 3 gets: 1, 4, 7, 10, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            node_id=1,
            node_count=3,
            batch_size=4,
        )
        manager.init_round(config)
        manager.start_generate()
        
        nonces = manager.get_next_nonces()
        assert nonces == [1, 4, 7, 10]
    
    def test_multi_node_nonces_node2(self, manager):
        """Node 2 of 3 gets: 2, 5, 8, 11, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            node_id=2,
            node_count=3,
            batch_size=4,
        )
        manager.init_round(config)
        manager.start_generate()
        
        nonces = manager.get_next_nonces()
        assert nonces == [2, 5, 8, 11]


class TestPoCManagerStatus:
    @pytest.fixture
    def mock_executor(self):
        return create_mock_model_executor()
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_get_status_idle(self, manager):
        status = manager.get_status()
        assert status["status"] == "IDLE"
        assert status["config"] is None
        assert status["stats"] is None
    
    def test_get_status_with_config(self, manager):
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            k_dim=12,
        )
        manager.init_round(config)
        
        status = manager.get_status()
        assert status["status"] == "IDLE"
        assert status["config"]["block_hash"] == "hash1"
        assert status["config"]["k_dim"] == 12
        assert status["stats"]["total_processed"] == 0


class TestPoCManagerBatch:
    """Test run_batch with mocked collective_rpc."""
    
    @pytest.fixture
    def mock_executor(self):
        executor = create_mock_model_executor()
        # Configure collective_rpc to return artifacts from "last PP rank"
        vectors = np.random.randn(4, 12).astype(np.float16)
        executor.collective_rpc.return_value = [
            None,  # First PP rank
            {  # Last PP rank
                "nonces": [0, 1, 2, 3],
                "vectors": vectors,
            }
        ]
        return executor
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_run_batch_calls_collective_rpc(self, manager, mock_executor):
        """run_batch should call collective_rpc with execute_poc_forward."""
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
            batch_size=4,
            seq_len=32,
        )
        manager.init_round(config)
        manager.start_generate()
        
        result = manager.run_batch()
        
        # Verify collective_rpc was called
        assert mock_executor.collective_rpc.call_count == 1
        from vllm.poc.poc_model_runner import execute_poc_forward
        last_call = mock_executor.collective_rpc.call_args_list[-1]
        assert last_call[0][0] == execute_poc_forward
        
        # Verify result contains artifacts
        assert result["should_continue"] is True
        assert len(result["artifacts"]) == 4
        assert result["artifacts"][0].nonce == 0
    
    def test_run_batch_updates_stats(self, manager, mock_executor):
        """run_batch should update stats."""
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
            batch_size=4,
            seq_len=32,
        )
        manager.init_round(config)
        manager.start_generate()
        
        assert manager.stats.total_processed == 0
        
        manager.run_batch()
        
        assert manager.stats.total_processed == 4
    
    def test_run_batch_returns_artifacts_with_base64(self, manager, mock_executor):
        """run_batch should return artifacts with base64 encoded vectors."""
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
            batch_size=4,
            seq_len=32,
            k_dim=12,
        )
        manager.init_round(config)
        manager.start_generate()
        
        result = manager.run_batch()
        
        # Verify artifacts have base64 encoded vectors
        for artifact in result["artifacts"]:
            assert artifact.nonce >= 0
            assert artifact.vector_b64  # Should be non-empty
            
            # Decode and verify shape
            vec = decode_vector(artifact.vector_b64)
            assert vec.shape == (12,)
    
    def test_run_batch_returns_empty_when_not_generating(self, manager):
        """run_batch returns empty result if not in GENERATING state."""
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
        )
        manager.init_round(config)
        # Don't call start_generate()
        
        result = manager.run_batch()
        
        assert result["should_continue"] is False


class TestPoCManagerGenerateArtifacts:
    """Test generate_artifacts for specific nonces."""
    
    @pytest.fixture
    def mock_executor(self):
        executor = create_mock_model_executor()
        vectors = np.random.randn(3, 12).astype(np.float16)
        executor.collective_rpc.return_value = [
            None,
            {
                "nonces": [5, 10, 15],
                "vectors": vectors,
            }
        ]
        return executor
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_generate_artifacts_for_specific_nonces(self, manager, mock_executor):
        """generate_artifacts should compute artifacts for specific nonces."""
        artifacts = manager.generate_artifacts(
            nonces=[5, 10, 15],
            block_hash="test_hash",
            public_key="test_node",
            seq_len=32,
            k_dim=12,
        )
        
        assert len(artifacts) == 3
        assert artifacts[0].nonce == 5
        assert artifacts[1].nonce == 10
        assert artifacts[2].nonce == 15
        
        # Verify vectors can be decoded
        for artifact in artifacts:
            vec = decode_vector(artifact.vector_b64)
            assert vec.shape == (12,)
    
    def test_generate_artifacts_updates_stats(self, manager, mock_executor):
        """generate_artifacts should update stats."""
        assert manager.stats.total_processed == 0
        
        manager.generate_artifacts(
            nonces=[5, 10, 15],
            block_hash="test_hash",
            public_key="test_node",
            seq_len=32,
            k_dim=12,
        )
        
        assert manager.stats.total_processed == 3


# GPU Tests - skipped for unit tests
@pytest.mark.gpu
@pytest.mark.skip(reason="GPU tests require full worker context - deferred to E2E tests")
class TestPoCManagerGPU:
    """GPU tests that load a real model."""
    pass

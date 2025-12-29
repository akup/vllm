import pytest
import time
import torch
from unittest.mock import Mock, patch, MagicMock

from vllm.poc.config import PoCConfig, PoCState
from vllm.poc.manager import PoCManager, PoCStats


class MockModelConfig:
    """Mock vLLM model config (v0 API)"""
    
    def get_hidden_size(self):
        return 128
    
    def get_vocab_size(self):
        return 1000


def create_mock_vllm_config():
    """Create a mock VllmConfig for testing.
    
    The manager uses vllm_config for set_forward_context(), which needs:
    - compilation_config.static_forward_context: dict of attention layers
    - parallel_config.data_parallel_size: for DP metadata (we set to 1 to skip)
    """
    vllm_config = MagicMock()
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.parallel_config.data_parallel_size = 1
    return vllm_config


class TestPoCStats:
    def test_initial_state(self):
        stats = PoCStats()
        assert stats.total_checked == 0
        assert stats.total_valid == 0
        assert stats.elapsed == 0.0
        assert stats.rate == 0.0
    
    def test_elapsed_calculation(self):
        stats = PoCStats(total_checked=100, start_time=time.time() - 10)
        assert stats.elapsed >= 10
        assert stats.elapsed < 11
    
    def test_rate_calculation(self):
        stats = PoCStats(total_checked=100, start_time=time.time() - 10)
        assert stats.rate >= 9
        assert stats.rate <= 11


class TestPoCManagerInit:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        param = torch.zeros(1)
        model.parameters.side_effect = lambda: iter([param])
        return model
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_model, mock_vllm_config):
        return PoCManager(mock_model, MockModelConfig(), mock_vllm_config)
    
    def test_initial_state(self, manager):
        assert manager.state == PoCState.IDLE
        assert manager.config is None
        assert manager.target is None
        assert manager.valid_nonces == []
        assert manager.valid_distances == []


class TestPoCManagerStateTransitions:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        param = torch.zeros(1, device="cpu")
        model.parameters.side_effect = lambda: iter([param])
        return model
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_model, mock_vllm_config):
        return PoCManager(mock_model, MockModelConfig(), mock_vllm_config)
    
    @pytest.fixture
    def config(self):
        return PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
        )
    
    def test_init_round_sets_config(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
        
        assert manager.state == PoCState.IDLE
        assert manager.config == config
    
    def test_start_generate_sets_state(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
        
        manager.start_generate()
        assert manager.state == PoCState.GENERATING
    
    def test_start_validate_sets_state(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
        
        manager.start_validate()
        assert manager.state == PoCState.VALIDATING
    
    def test_generating_to_validating_transition(self, manager, config):
        """Test GENERATING -> VALIDATING transition."""
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
        
        manager.start_generate()
        assert manager.state == PoCState.GENERATING
        
        manager.start_validate()
        assert manager.state == PoCState.VALIDATING
    
    def test_init_round_resets_counters(self, manager, config):
        manager.valid_nonces = [1, 2, 3]
        manager.valid_distances = [0.1, 0.2, 0.3]
        manager.stats.total_checked = 100
        
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
        
        assert manager.valid_nonces == []
        assert manager.valid_distances == []
        assert manager.stats.total_checked == 0
    
    def test_init_round_raises_if_already_generating(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
            manager.start_generate()
        
        with pytest.raises(RuntimeError, match="Round already in progress"):
            manager.init_round(config)
    
    def test_start_generate_requires_init(self, manager):
        with pytest.raises(RuntimeError, match="Round not initialized"):
            manager.start_generate()
    
    def test_start_validate_requires_init(self, manager):
        with pytest.raises(RuntimeError, match="Round not initialized"):
            manager.start_validate()
    
    def test_stop_round_from_generating(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
            manager.start_generate()
        
        manager.stop_round()
        assert manager.state == PoCState.STOPPED
    
    def test_stop_round_from_validating(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
            manager.start_validate()
        
        manager.stop_round()
        assert manager.state == PoCState.STOPPED


class TestPoCManagerNonceGeneration:
    """Cross-check: Nonce iteration pattern with original NonceIterator"""
    
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        param = torch.zeros(1, device="cpu")
        model.parameters.side_effect = lambda: iter([param])
        return model
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_model, mock_vllm_config):
        return PoCManager(mock_model, MockModelConfig(), mock_vllm_config)
    
    def test_single_node_nonces(self, manager):
        """Single node gets sequential nonces: 0, 1, 2, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            node_id=0,
            node_count=1,
            batch_size=4,
        )
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
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
            r_target=0.5,
            node_id=0,
            node_count=3,
            batch_size=4,
        )
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
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
            r_target=0.5,
            node_id=1,
            node_count=3,
            batch_size=4,
        )
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
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
            r_target=0.5,
            node_id=2,
            node_count=3,
            batch_size=4,
        )
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
            manager.start_generate()
        
        nonces = manager.get_next_nonces()
        assert nonces == [2, 5, 8, 11]


class TestPoCManagerStatus:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        param = torch.zeros(1, device="cpu")
        model.parameters.side_effect = lambda: iter([param])
        return model
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_model, mock_vllm_config):
        return PoCManager(mock_model, MockModelConfig(), mock_vllm_config)
    
    def test_get_status_idle(self, manager):
        status = manager.get_status()
        assert status["state"] == "IDLE"
        assert status["valid_nonces"] == []
        assert status["valid_distances"] == []
        assert status["total_checked"] == 0
        assert status["total_valid"] == 0
    
    def test_get_status_with_data(self, manager):
        manager.valid_nonces = [1, 2, 3]
        manager.valid_distances = [0.1, 0.2, 0.3]
        manager.stats.total_checked = 100
        manager.stats.total_valid = 3
        
        status = manager.get_status()
        assert status["valid_nonces"] == [1, 2, 3]
        assert status["valid_distances"] == [0.1, 0.2, 0.3]
        assert status["total_checked"] == 100
        assert status["total_valid"] == 3


# GPU Tests - require CUDA and load actual model
# TODO: Update for vLLM v1 - model access path differs in v1 engine
# TODO: These tests are skipped because running the model outside the worker
#       context causes parallel group initialization errors. Proper GPU tests
#       should be done in Phase 5 E2E tests where the full vLLM server runs.
@pytest.mark.gpu
@pytest.mark.skip(reason="GPU tests require full worker context - deferred to Phase 5 E2E tests")
class TestPoCManagerGPU:
    """GPU tests that load a real model to test run_batch and validate.
    
    Note: Currently uses vLLM v0 API for model access.
    TODO: Update model access pattern for vLLM v1 compatibility.
    
    WARNING: These tests are currently skipped because:
    - The model needs to run inside the vLLM worker process context
    - Extracting the model and running it directly causes parallel group errors
    - Proper testing should be done in Phase 5 with full E2E integration tests
    """
    
    @pytest.fixture(scope="class")
    def model_and_config(self):
        """Load Qwen3-0.6B model for testing (v0 API)."""
        import os
        from vllm import LLM
        
        # Force v0 engine
        os.environ["VLLM_USE_V1"] = "0"
        
        llm = LLM(
            model="Qwen/Qwen3-0.6B",
            enforce_eager=True,
            gpu_memory_utilization=0.3,
            max_model_len=1024,
        )
        # v0 API: access model via model_executor.driver_worker.model_runner.model
        # TODO: For v1, use worker.get_model() or collective_rpc pattern
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        model_config = llm.llm_engine.model_config
        vllm_config = llm.llm_engine.vllm_config
        
        yield model, model_config, vllm_config
        
        del llm
    
    def test_run_batch_produces_valid_distances(self, model_and_config):
        """run_batch produces ProofBatch with distances in [0, 2]."""
        model, model_config, vllm_config = model_and_config
        manager = PoCManager(model, model_config, vllm_config)
        
        config = PoCConfig(
            block_hash="test_block_hash",
            block_height=100,
            public_key="test_node",
            r_target=0.5,
            batch_size=4,
            seq_len=32,
        )
        manager.init_round(config)
        manager.start_generate()
        
        batch = manager.run_batch()
        
        assert len(batch) == 4
        assert all(0 <= d <= 2 for d in batch.dist)
        assert batch.block_hash == "test_block_hash"
        assert batch.public_key == "test_node"
    
    def test_run_batch_determinism(self, model_and_config):
        """Same inputs produce same outputs."""
        model, model_config, vllm_config = model_and_config
        
        manager1 = PoCManager(model, model_config, vllm_config)
        config1 = PoCConfig(
            block_hash="determinism_test",
            block_height=100,
            public_key="test_node",
            r_target=0.5,
            batch_size=4,
            seq_len=32,
        )
        manager1.init_round(config1)
        manager1.start_generate()
        batch1 = manager1.run_batch()
        
        manager2 = PoCManager(model, model_config, vllm_config)
        config2 = PoCConfig(
            block_hash="determinism_test",
            block_height=100,
            public_key="test_node",
            r_target=0.5,
            batch_size=4,
            seq_len=32,
        )
        manager2.init_round(config2)
        manager2.start_generate()
        batch2 = manager2.run_batch()
        
        assert batch1.nonces == batch2.nonces
        for d1, d2 in zip(batch1.dist, batch2.dist):
            assert abs(d1 - d2) < 1e-5
    
    def test_validate_recomputes_same_distances(self, model_and_config):
        """validate() recomputes same distances for same nonces."""
        model, model_config, vllm_config = model_and_config
        manager = PoCManager(model, model_config, vllm_config)
        
        config = PoCConfig(
            block_hash="validate_test",
            block_height=100,
            public_key="test_node",
            r_target=0.5,
            batch_size=4,
            seq_len=32,
        )
        manager.init_round(config)
        manager.start_generate()
        
        batch = manager.run_batch()
        
        manager.start_validate()
        distances, valid = manager.validate(batch.nonces, batch.public_key)
        
        for orig, recomputed in zip(batch.dist, distances):
            assert abs(orig - recomputed) < 1e-5
    
    def test_different_public_key_different_distances(self, model_and_config):
        """Different public_key produces different distances."""
        model, model_config, vllm_config = model_and_config
        manager = PoCManager(model, model_config, vllm_config)
        
        config = PoCConfig(
            block_hash="pubkey_test",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            batch_size=4,
            seq_len=32,
        )
        manager.init_round(config)
        manager.start_generate()
        batch = manager.run_batch()
        
        manager.start_validate()
        distances_other, _ = manager.validate(batch.nonces, "node2")
        
        assert batch.dist != distances_other
    
    def test_stats_updated_after_batch(self, model_and_config):
        """Stats are updated after run_batch."""
        model, model_config, vllm_config = model_and_config
        manager = PoCManager(model, model_config, vllm_config)
        
        config = PoCConfig(
            block_hash="stats_test",
            block_height=100,
            public_key="test_node",
            r_target=2.0,
            batch_size=4,
            seq_len=32,
        )
        manager.init_round(config)
        manager.start_generate()
        
        assert manager.stats.total_checked == 0
        
        manager.run_batch()
        
        assert manager.stats.total_checked == 4


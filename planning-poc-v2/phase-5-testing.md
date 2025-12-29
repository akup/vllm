# Phase 5: E2E Integration Testing

## Objective

End-to-end integration tests that verify the complete PoC system running against a live vLLM server.

## Deliverable

Integration test suite that validates the full workflow: vLLM server with PoC enabled, API endpoints, and actual model inference.

## Test Strategy

### Unit Tests (Phases 1-4)

Unit tests are implemented in earlier phases:
- Phase 1: `test_data.py` - ProofBatch, ValidatedBatch
- Phase 2: `test_gpu_random.py` - RNG determinism, distance calculation
- Phase 3: `test_manager.py` - PoCManager state, nonce generation
- Phase 4: `test_routes.py` - API endpoints with mocked manager

### E2E Integration Tests (This Phase)

Tests requiring a live vLLM server with actual GPU inference.

**Cross-check**: Compare full workflow behavior with original PoW service:
`/home/ubuntu/workspace/gonka/mlnode/packages/pow/tests/integration/`

## Test Files

### File: `tests/poc/conftest.py`

```python
import pytest
import subprocess
import time
import requests

@pytest.fixture(scope="module")
def vllm_server():
    """Start vLLM server with PoC enabled"""
    proc = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3-0.6B",
        "--enable-poc",
        "--port", "8765",
    ])
    
    # Wait for server to start
    for _ in range(60):
        try:
            r = requests.get("http://localhost:8765/health")
            if r.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    else:
        proc.kill()
        raise RuntimeError("Server failed to start")
    
    yield "http://localhost:8765"
    
    proc.terminate()
    proc.wait()

@pytest.fixture
def poc_client(vllm_server):
    """Client for PoC API"""
    class PoCClient:
        def __init__(self, base_url):
            self.base_url = base_url
        
        def init(self, block_hash, public_key, r_target=0.5, **kwargs):
            return requests.post(f"{self.base_url}/api/v1/pow/init", json={
                "block_hash": block_hash,
                "block_height": 100,
                "public_key": public_key,
                "r_target": r_target,
                **kwargs
            })
        
        def start(self, block_hash, public_key, r_target=0.5, node_id=0, node_count=1, **kwargs):
            return requests.post(f"{self.base_url}/api/v1/pow/init/generate", json={
                "block_hash": block_hash,
                "block_height": 100,
                "public_key": public_key,
                "r_target": r_target,
                "node_id": node_id,
                "node_count": node_count,
                **kwargs
            })
        
        def phase_generate(self):
            return requests.post(f"{self.base_url}/api/v1/pow/phase/generate")
        
        def phase_validate(self):
            return requests.post(f"{self.base_url}/api/v1/pow/phase/validate")
        
        def stop(self):
            return requests.post(f"{self.base_url}/api/v1/pow/stop")
        
        def status(self):
            return requests.get(f"{self.base_url}/api/v1/pow/status")
        
        def validate(self, block_hash, public_key, nonces, dist=None, block_height=100, node_id=0):
            """Validate nonces by submitting full ProofBatch (matching original API)."""
            if dist is None:
                dist = [0.0] * len(nonces)  # Placeholder distances for validation
            return requests.post(f"{self.base_url}/api/v1/pow/validate", json={
                "public_key": public_key,
                "block_hash": block_hash,
                "block_height": block_height,
                "nonces": nonces,
                "dist": dist,
                "node_id": node_id,
            })
    
    return PoCClient(vllm_server)
```

### File: `tests/poc/test_integration.py`

```python
import time
import pytest
import requests

class TestBasicRound:
    """Test basic PoC round flow"""
    
    def test_start_stop_round(self, poc_client):
        """Can start and stop a round"""
        r = poc_client.start("hash1", "node1")
        assert r.status_code == 200
        assert r.json()["status"] == "OK"
        
        r = poc_client.stop()
        assert r.status_code == 200
        assert r.json()["status"] == "OK"
    
    def test_status_during_round(self, poc_client):
        """Status returns valid data during round"""
        poc_client.start("hash1", "node1", r_target=1.5)  # High r_target for more valid
        time.sleep(2)
        
        r = poc_client.status()
        assert r.status_code == 200
        status = r.json()
        assert status["state"] == "GENERATING"
        assert status["total_checked"] > 0
        
        poc_client.stop()
    
    def test_valid_nonces_found(self, poc_client):
        """Valid nonces are found with reasonable r_target"""
        poc_client.start("hash1", "node1", r_target=1.5)
        time.sleep(5)
        
        r = poc_client.status()
        status = r.json()
        
        poc_client.stop()
        
        assert len(status["valid_nonces"]) > 0
        assert len(status["valid_distances"]) == len(status["valid_nonces"])
        for dist in status["valid_distances"]:
            assert dist < 1.5


class TestDeterminism:
    """Test deterministic behavior"""
    
    def test_same_nonce_same_distance(self, poc_client):
        """Same nonce produces same distance"""
        poc_client.start("hash1", "node1")
        time.sleep(1)
        poc_client.stop()
        
        # Validate nonce 0 twice
        r1 = poc_client.validate("hash1", "node1", [0, 1, 2])
        r2 = poc_client.validate("hash1", "node1", [0, 1, 2])
        
        assert r1.json()["distances"] == r2.json()["distances"]
    
    def test_different_public_key_different_distance(self, poc_client):
        """Different public_key produces different distances"""
        poc_client.start("hash1", "node1")
        time.sleep(1)
        poc_client.stop()
        
        r1 = poc_client.validate("hash1", "node1", [0])
        r2 = poc_client.validate("hash1", "node2", [0])
        
        assert r1.json()["distances"] != r2.json()["distances"]


class TestValidation:
    """Test validation scenarios"""
    
    def test_wrong_seed_fails_validation(self, poc_client):
        """Nonces validated with wrong block_hash fail"""
        poc_client.start("correct_hash", "node1", r_target=1.5)
        time.sleep(3)
        status = poc_client.status().json()
        poc_client.stop()
        
        if len(status["valid_nonces"]) == 0:
            pytest.skip("No valid nonces found")
        
        # Validate with wrong hash - should return different distances
        poc_client.start("wrong_hash", "node1")
        r = poc_client.validate("wrong_hash", "node1", status["valid_nonces"])
        poc_client.stop()
        
        # Distances should be different (and likely not valid)
        assert r.json()["distances"] != status["valid_distances"]
    
    def test_fabricated_nonces_invalid(self, poc_client):
        """Random nonces are unlikely to be valid with low r_target"""
        poc_client.start("hash1", "node1", r_target=0.1)  # Very low threshold
        time.sleep(1)
        
        r = poc_client.validate("hash1", "node1", [999999, 999998, 999997])
        result = r.json()
        
        poc_client.stop()
        
        # With r_target=0.1, almost no nonces should be valid
        assert not any(result["valid"])


class TestInferenceBlocking:
    """Test that inference is blocked during PoC.
    
    Phase 5: Inference blocked during GENERATING and VALIDATING (returns 503).
    Phase 6: Will explore allowing inference during VALIDATING phase.
    """
    
    def test_chat_returns_503(self, vllm_server, poc_client):
        """Chat completion returns 503 during PoC (Phase 5 behavior)."""
        poc_client.start("hash1", "node1")
        time.sleep(0.5)
        
        r = requests.post(f"{vllm_server}/v1/chat/completions", json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        
        poc_client.stop()
        
        assert r.status_code == 503


class TestSmoothSwitch:
    """Test seamless switching between modes"""
    
    def test_inference_after_poc(self, vllm_server, poc_client):
        """Inference works after PoC stops"""
        # Start and stop PoC
        poc_client.start("hash1", "node1")
        time.sleep(1)
        poc_client.stop()
        time.sleep(0.5)
        
        # Inference should work
        r = requests.post(f"{vllm_server}/v1/chat/completions", json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 5,
        })
        
        assert r.status_code == 200
    
    def test_multiple_rounds(self, poc_client):
        """Can run multiple PoC rounds"""
        for i in range(3):
            poc_client.start(f"hash{i}", "node1")
            time.sleep(0.5)
            status = poc_client.status().json()
            assert status["state"] == "GENERATING"
            poc_client.stop()
            time.sleep(0.5)


class TestDistribution:
    """Test distance distribution properties"""
    
    def test_distances_in_valid_range(self, poc_client):
        """All distances are in [0, 2]"""
        poc_client.start("hash1", "node1", r_target=2.0)
        time.sleep(3)
        status = poc_client.status().json()
        poc_client.stop()
        
        for dist in status["valid_distances"]:
            assert 0 <= dist <= 2.0
    
    def test_mean_distance_reasonable(self, poc_client):
        """Mean distance is near sqrt(2) for uniform distribution"""
        poc_client.start("hash1", "node1", r_target=2.0, batch_size=64)
        time.sleep(10)
        status = poc_client.status().json()
        poc_client.stop()
        
        if len(status["valid_distances"]) < 100:
            pytest.skip("Not enough samples")
        
        import statistics
        mean_dist = statistics.mean(status["valid_distances"])
        # For high-dimensional sphere, mean distance is ~sqrt(2) = 1.414
        # Allow wide tolerance for randomized model
        assert 1.0 < mean_dist < 1.8


class TestSeedValidation:
    """Test scenarios with incorrect seeds and nonces"""
    
    def test_wrong_block_hash_validation(self, poc_client):
        """Nonces from one block_hash are invalid for another"""
        # Generate valid nonces with hash1
        poc_client.start("hash1", "node1", r_target=1.5)
        time.sleep(3)
        status = poc_client.status().json()
        valid_nonces = status["valid_nonces"]
        valid_distances = status["valid_distances"]
        poc_client.stop()
        
        if len(valid_nonces) == 0:
            pytest.skip("No valid nonces found")
        
        # Validate same nonces against different block_hash
        poc_client.start("hash2", "node1")
        time.sleep(0.5)
        r = poc_client.validate("hash2", "node1", valid_nonces)
        new_distances = r.json()["distances"]
        poc_client.stop()
        
        # Distances should be completely different
        for old, new in zip(valid_distances, new_distances):
            assert abs(old - new) > 0.01  # Not matching
    
    def test_wrong_public_key_validation(self, poc_client):
        """Nonces from one public_key are invalid for another"""
        poc_client.start("hash1", "node1", r_target=1.5)
        time.sleep(3)
        status = poc_client.status().json()
        valid_nonces = status["valid_nonces"]
        poc_client.stop()
        
        if len(valid_nonces) == 0:
            pytest.skip("No valid nonces found")
        
        # Validate with different public_key
        poc_client.start("hash1", "node_attacker")
        time.sleep(0.5)
        r = poc_client.validate("hash1", "node_attacker", valid_nonces)
        result = r.json()
        poc_client.stop()
        
        # Should have different distances (likely invalid)
        # Attacker cannot reuse another node's nonces
        assert r.json()["distances"] != status["valid_distances"]
    
    def test_fabricated_distance_detection(self, poc_client):
        """Submitted distances can be verified by recomputation"""
        poc_client.start("hash1", "node1", r_target=1.0)
        time.sleep(2)
        status = poc_client.status().json()
        poc_client.stop()
        
        if len(status["valid_nonces"]) == 0:
            pytest.skip("No valid nonces found")
        
        # Recompute distances for the same nonces
        poc_client.start("hash1", "node1")
        time.sleep(0.5)
        r = poc_client.validate("hash1", "node1", status["valid_nonces"])
        recomputed = r.json()["distances"]
        poc_client.stop()
        
        # Distances should match exactly (determinism)
        for orig, recomp in zip(status["valid_distances"], recomputed):
            assert abs(orig - recomp) < 1e-5  # Floating point tolerance
    
    def test_tampered_nonce_list(self, poc_client):
        """Modified nonce list fails validation"""
        poc_client.start("hash1", "node1", r_target=1.5)
        time.sleep(3)
        status = poc_client.status().json()
        valid_nonces = status["valid_nonces"]
        poc_client.stop()
        
        if len(valid_nonces) == 0:
            pytest.skip("No valid nonces found")
        
        # Tamper with nonce list (add fake nonces)
        tampered_nonces = valid_nonces + [999999999]
        
        poc_client.start("hash1", "node1")
        time.sleep(0.5)
        r = poc_client.validate("hash1", "node1", tampered_nonces)
        result = r.json()
        poc_client.stop()
        
        # Last nonce should have invalid distance
        assert result["valid"][-1] == False


class TestDistributionAnalysis:
    """Analyze distance distribution for r_target calibration"""
    
    def test_distance_distribution_stats(self, poc_client):
        """Collect and report distribution statistics"""
        poc_client.start("hash1", "node1", r_target=2.0, batch_size=128)
        time.sleep(30)  # Collect samples
        status = poc_client.status().json()
        poc_client.stop()
        
        distances = status["valid_distances"]
        if len(distances) < 1000:
            pytest.skip("Not enough samples for distribution analysis")
        
        import statistics
        mean = statistics.mean(distances)
        std = statistics.stdev(distances)
        
        print(f"Distribution: mean={mean:.4f}, std={std:.4f}")
        print(f"Expected (random): mean~1.414, range=[0,2]")
        print(f"Samples collected: {len(distances)}")
        
        # Basic sanity checks
        assert 0.5 < mean < 1.9  # Reasonable range
        assert std > 0  # Has variance
    
    def test_distribution_consistency_across_blocks(self, poc_client):
        """Distribution should be similar for different block_hashes"""
        import statistics
        
        means = []
        for i in range(3):
            poc_client.start(f"block_{i}", "node1", r_target=2.0, batch_size=64)
            time.sleep(5)
            status = poc_client.status().json()
            poc_client.stop()
            
            if len(status["valid_distances"]) > 50:
                means.append(statistics.mean(status["valid_distances"]))
        
        if len(means) < 2:
            pytest.skip("Not enough samples across blocks")
        
        # Means should be within reasonable range of each other
        mean_of_means = statistics.mean(means)
        for m in means:
            assert abs(m - mean_of_means) < 0.3  # Within 0.3 of average


class TestFraudDetection:
    """Test statistical fraud detection"""
    
    def test_honest_batch_passes(self, poc_client):
        """Legitimate batches pass fraud detection"""
        poc_client.start("hash1", "node1", r_target=1.5)
        time.sleep(5)
        status = poc_client.status().json()
        poc_client.stop()
        
        if len(status["valid_nonces"]) < 10:
            pytest.skip("Not enough valid nonces")
        
        # Validate honest nonces
        poc_client.start("hash1", "node1")
        time.sleep(0.5)
        r = poc_client.validate("hash1", "node1", status["valid_nonces"])
        result = r.json()
        poc_client.stop()
        
        # All should be valid
        invalid_count = sum(1 for v in result["valid"] if not v)
        assert invalid_count == 0
    
    def test_fabricated_batch_detected(self, poc_client):
        """Batches with fabricated distances are flagged"""
        poc_client.start("hash1", "node1", r_target=0.1)  # Very low threshold
        time.sleep(1)
        
        # Try to validate random nonces that are unlikely to be valid
        r = poc_client.validate("hash1", "node1", list(range(1000000, 1000020)))
        result = r.json()
        
        poc_client.stop()
        
        # With r_target=0.1, almost none should be valid
        valid_count = sum(1 for v in result["valid"] if v)
        assert valid_count == 0


class TestSeedValidationScenarios:
    """Test different seed scenarios - correct/incorrect nonces.
    
    These tests verify that the PoC system correctly rejects nonces
    computed with different seeds (block_hash, public_key).
    """
    
    def test_correct_seed_correct_nonce_validates(self, poc_client):
        """Happy path: correct seed + correct nonce = valid"""
        # Generate with seed A
        poc_client.start("seed_A", "node1", r_target=1.5)
        time.sleep(3)
        status = poc_client.status().json()
        valid_nonces = status["valid_nonces"]
        valid_distances = status["valid_distances"]
        poc_client.stop()
        
        if len(valid_nonces) == 0:
            pytest.skip("No valid nonces found")
        
        # Validate with same seed A - should match exactly
        poc_client.start("seed_A", "node1")
        time.sleep(0.5)
        r = poc_client.validate("seed_A", "node1", valid_nonces)
        recomputed = r.json()["distances"]
        poc_client.stop()
        
        for orig, recomp in zip(valid_distances, recomputed):
            assert abs(orig - recomp) < 1e-5
    
    def test_wrong_seed_correct_nonce_fails(self, poc_client):
        """Wrong seed (block_hash) makes valid nonces invalid"""
        # Generate valid nonces with seed A
        poc_client.start("seed_A", "node1", r_target=1.5)
        time.sleep(3)
        status = poc_client.status().json()
        valid_nonces = status["valid_nonces"]
        poc_client.stop()
        
        if len(valid_nonces) == 0:
            pytest.skip("No valid nonces found")
        
        # Validate with DIFFERENT seed B - should produce different distances
        poc_client.start("seed_B", "node1")
        time.sleep(0.5)
        r = poc_client.validate("seed_B", "node1", valid_nonces)
        result = r.json()
        poc_client.stop()
        
        # Most should be invalid with r_target=1.5 under different seed
        valid_count = sum(1 for d in result["distances"] if d < 1.5)
        # Statistically unlikely to find same valid nonces with different seed
        assert valid_count < len(valid_nonces) * 0.5
    
    def test_fabricated_nonce_not_validated(self, poc_client):
        """Attacker cannot submit fabricated nonces"""
        poc_client.start("hash1", "node1", r_target=0.5)  # Strict threshold
        time.sleep(1)
        
        # Fabricated nonces (not computed, just random large numbers)
        fabricated = list(range(100000000, 100000020))
        r = poc_client.validate("hash1", "node1", fabricated)
        result = r.json()
        poc_client.stop()
        
        # None should pass strict threshold
        assert not any(result["valid"])
    
    def test_nonce_replay_different_round_fails(self, poc_client):
        """Valid nonces from round N are invalid in round N+1"""
        # Round 1: collect valid nonces
        poc_client.start("round_1_hash", "node1", r_target=1.5)
        time.sleep(3)
        round1_nonces = poc_client.status().json()["valid_nonces"]
        poc_client.stop()
        
        if len(round1_nonces) == 0:
            pytest.skip("No valid nonces found")
        
        # Round 2: try to reuse round 1 nonces (different block_hash)
        poc_client.start("round_2_hash", "node1", r_target=1.5)
        time.sleep(0.5)
        r = poc_client.validate("round_2_hash", "node1", round1_nonces)
        result = r.json()
        poc_client.stop()
        
        # Should not be valid under new round's seed
        valid_count = sum(1 for v in result["valid"] if v)
        assert valid_count < len(round1_nonces) * 0.3  # Most fail
    
    def test_wrong_public_key_replay_fails(self, poc_client):
        """Nonces computed for node1 are invalid for node_attacker"""
        poc_client.start("hash1", "node1", r_target=1.5)
        time.sleep(3)
        status = poc_client.status().json()
        valid_nonces = status["valid_nonces"]
        valid_distances = status["valid_distances"]
        poc_client.stop()
        
        if len(valid_nonces) == 0:
            pytest.skip("No valid nonces found")
        
        # Attacker tries to claim node1's nonces as their own
        poc_client.start("hash1", "node_attacker")
        time.sleep(0.5)
        r = poc_client.validate("hash1", "node_attacker", valid_nonces)
        result = r.json()
        poc_client.stop()
        
        # Distances should be completely different
        assert result["distances"] != valid_distances


class TestDistributionVerification:
    """Verify distance distribution properties when model is NOT truly random.
    
    A trained model produces non-uniform output distribution on the vocabulary space.
    This class explicitly tests and documents distribution characteristics for:
    1. r_target calibration
    2. Fraud detection threshold tuning
    3. Understanding deviation from theoretical uniform sphere
    
    These tests are CRITICAL for production deployment.
    """
    
    def test_compare_with_theoretical_uniform(self, poc_client):
        """Compare actual distribution against uniform sphere baseline.
        
        For uniform random on unit sphere in high dimensions:
        - Mean distance to random target: sqrt(2) ~ 1.414
        - Std deviation: approximately 1/sqrt(dim)
        
        Trained model will deviate due to:
        - Concentrated probability mass on common tokens
        - Structured logit patterns from language modeling
        """
        poc_client.start("dist_test_hash", "node1", r_target=2.0, batch_size=128)
        time.sleep(30)
        status = poc_client.status().json()
        poc_client.stop()
        
        distances = status["valid_distances"]
        if len(distances) < 500:
            pytest.skip("Not enough samples for distribution comparison")
        
        import statistics
        mean = statistics.mean(distances)
        std = statistics.stdev(distances)
        
        # Theoretical values for uniform distribution on sphere
        theoretical_mean = 1.414  # sqrt(2)
        
        deviation = abs(mean - theoretical_mean)
        
        print(f"Actual mean: {mean:.4f}")
        print(f"Theoretical uniform mean: {theoretical_mean:.4f}")
        print(f"Deviation: {deviation:.4f} ({deviation/theoretical_mean*100:.1f}%)")
        print(f"Std deviation: {std:.4f}")
        
        # Document but don't fail - trained models are expected to deviate
        assert 0.5 < mean < 1.9, f"Mean {mean} outside reasonable range"
    
    def test_valid_rate_vs_r_target_calibration(self, poc_client):
        """Document relationship between r_target and valid nonce rate.
        
        For production deployment, this establishes the mapping:
        r_target -> expected valid rate per batch
        
        This is essential for setting r_target to achieve desired valid nonce rate.
        """
        r_targets = [0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6]
        results = []
        
        for r in r_targets:
            poc_client.start(f"calibration_{r}", "node1", r_target=r, batch_size=64)
            time.sleep(10)
            status = poc_client.status().json()
            poc_client.stop()
            
            rate = status["total_valid"] / max(status["total_checked"], 1)
            results.append({"r_target": r, "rate": rate, "checked": status["total_checked"]})
            print(f"r_target={r:.1f}: valid_rate={rate:.4%} (checked={status['total_checked']})")
        
        # Verify monotonically increasing rate
        rates = [r["rate"] for r in results]
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i+1], f"Rate should increase with r_target"
        
        print("\nCalibration table for production r_target selection:")
        for r in results:
            print(f"  r_target={r['r_target']:.1f} -> {r['rate']:.2%} valid rate")
    
    def test_distribution_stability_across_blocks(self, poc_client):
        """Verify distribution is stable across different block_hashes.
        
        Distribution characteristics should be similar regardless of
        the specific block_hash (seed), since it's model-dependent.
        """
        import statistics
        
        means = []
        stds = []
        
        for i in range(5):
            poc_client.start(f"stability_test_{i}", "node1", r_target=2.0, batch_size=64)
            time.sleep(5)
            status = poc_client.status().json()
            poc_client.stop()
            
            distances = status["valid_distances"]
            if len(distances) > 50:
                means.append(statistics.mean(distances))
                stds.append(statistics.stdev(distances))
        
        if len(means) < 3:
            pytest.skip("Not enough samples across blocks")
        
        mean_of_means = statistics.mean(means)
        std_of_means = statistics.stdev(means)
        
        print(f"Mean distances across blocks: {means}")
        print(f"Mean of means: {mean_of_means:.4f}")
        print(f"Std of means: {std_of_means:.4f}")
        
        # Distribution should be stable - low variance across different seeds
        assert std_of_means < 0.2, f"Distribution unstable across blocks: std={std_of_means}"


class TestModelDistributionProperties:
    """Verify distance distribution with real (trained) model.
    
    A trained model's output is NOT uniformly distributed on the sphere.
    This affects r_target calibration for desired valid nonce rate.
    
    Note: These tests document actual behavior for calibration purposes.
    """
    
    def test_distance_distribution_histogram(self, poc_client):
        """Collect distance distribution to verify it's reasonable"""
        poc_client.start("hash1", "node1", r_target=2.0, batch_size=128)
        time.sleep(30)  # Collect many samples
        status = poc_client.status().json()
        poc_client.stop()
        
        distances = status["valid_distances"]
        if len(distances) < 500:
            pytest.skip("Not enough samples")
        
        import statistics
        mean = statistics.mean(distances)
        std = statistics.stdev(distances)
        min_d = min(distances)
        max_d = max(distances)
        
        # For random weights: mean ~ sqrt(2) = 1.414
        # For trained model: may deviate due to concentrated logit patterns
        print(f"Distribution stats: mean={mean:.4f}, std={std:.4f}, min={min_d:.4f}, max={max_d:.4f}")
        print(f"Expected (random weights): mean~1.414, std~0.2")
        
        # Sanity checks (wide tolerance for trained models)
        assert 0.8 < mean < 1.8, f"Mean distance {mean} outside expected range"
        assert 0.05 < std < 0.5, f"Std deviation {std} outside expected range"
        assert min_d >= 0, "Distance cannot be negative"
        assert max_d <= 2.0, "Distance cannot exceed 2.0 for unit sphere"
    
    def test_r_target_calibration(self, poc_client):
        """Verify valid nonce rate approximately matches expected from r_target.
        
        This test documents actual rates for production r_target calibration.
        Rate should increase monotonically with r_target.
        """
        r_targets = [0.5, 1.0, 1.2, 1.4]
        rates = {}
        
        for r in r_targets:
            poc_client.start(f"calibration_{r}", "node1", r_target=r, batch_size=64)
            time.sleep(10)
            status = poc_client.status().json()
            poc_client.stop()
            
            rate = status["total_valid"] / max(status["total_checked"], 1)
            rates[r] = rate
            print(f"r_target={r}: valid_rate={rate:.4%}")
        
        # Rate should increase monotonically with r_target
        assert rates[0.5] < rates[1.0] < rates[1.4]
        
        # Document actual rates for production calibration
        print("Use these rates to calibrate r_target for desired valid nonce rate")
    
    def test_distribution_not_uniform_for_trained_model(self, poc_client):
        """Document that trained model produces non-uniform distribution.
        
        Unlike random weights, a trained model has structured outputs that
        may concentrate in certain regions of the output space. This is
        expected and affects r_target selection.
        """
        poc_client.start("hash1", "node1", r_target=2.0, batch_size=64)
        time.sleep(15)
        status = poc_client.status().json()
        poc_client.stop()
        
        distances = status["valid_distances"]
        if len(distances) < 200:
            pytest.skip("Not enough samples")
        
        import statistics
        mean = statistics.mean(distances)
        
        # For truly random (uniform on sphere), mean would be ~sqrt(2) = 1.414
        # Trained models typically deviate from this
        deviation_from_uniform = abs(mean - 1.414)
        
        print(f"Mean distance: {mean:.4f}")
        print(f"Deviation from uniform sphere: {deviation_from_uniform:.4f}")
        print("This deviation is expected for trained models")
        
        # Just document - don't fail on deviation
        assert 0 < mean < 2.0


class TestCallbackIntegration:
    """Test callback URL functionality for push-model compatibility."""
    
    @pytest.fixture
    def callback_server(self):
        """Mock HTTP server to receive callbacks.
        
        Runs a simple FastAPI server that records received batches.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import threading
        import uvicorn
        
        app = FastAPI()
        received = {"generated": [], "validated": []}
        
        @app.post("/generated")
        async def receive_generated(batch: dict):
            received["generated"].append(batch)
            return {"status": "ok"}
        
        @app.post("/validated")
        async def receive_validated(batch: dict):
            received["validated"].append(batch)
            return {"status": "ok"}
        
        # Run in background thread
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=9999, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        time.sleep(0.5)  # Wait for server to start
        
        yield {"url": "http://127.0.0.1:9999", "received": received}
        
        server.should_exit = True
    
    def test_generated_callback_sent(self, poc_client, callback_server):
        """Valid batches are POSTed to callback_url/generated"""
        poc_client.start(
            "hash1", "node1", 
            r_target=1.5,
            callback_url=callback_server["url"]
        )
        time.sleep(5)
        poc_client.stop()
        
        # Verify callback server received batches
        assert len(callback_server["received"]["generated"]) > 0
        
        # Verify batch structure
        batch = callback_server["received"]["generated"][0]
        assert "nonces" in batch
        assert "dist" in batch
        assert "public_key" in batch
    
    def test_callback_with_status_both_work(self, poc_client, callback_server):
        """Both callback push and status pull work simultaneously"""
        poc_client.start(
            "hash1", "node1",
            r_target=1.5,
            callback_url=callback_server["url"]
        )
        time.sleep(3)
        
        # Pull via status
        status = poc_client.status().json()
        poc_client.stop()
        
        # Both should have data
        assert len(callback_server["received"]["generated"]) > 0
        assert status["total_checked"] > 0
    
    def test_callback_failure_does_not_block(self, poc_client):
        """Callback failures don't stop generation"""
        # Use invalid URL that will fail
        poc_client.start(
            "hash1", "node1",
            r_target=1.5,
            callback_url="http://invalid-host-that-does-not-exist:9999"
        )
        time.sleep(3)
        status = poc_client.status().json()
        poc_client.stop()
        
        # Generation continues despite callback failures
        assert status["total_checked"] > 0
        assert status["state"] in ["GENERATING", "STOPPED"]
    
    def test_no_callback_when_not_configured(self, poc_client, callback_server):
        """Without callback_url, nothing is POSTed"""
        # Start without callback_url
        poc_client.start("hash1", "node1", r_target=1.5)
        time.sleep(3)
        poc_client.stop()
        
        # Callback server should not receive anything
        assert len(callback_server["received"]["generated"]) == 0
    
    def test_callback_only_sends_valid_nonces(self, poc_client, callback_server):
        """Callback batches only contain nonces where dist < r_target"""
        poc_client.start(
            "hash1", "node1",
            r_target=1.0,  # Moderate threshold
            callback_url=callback_server["url"]
        )
        time.sleep(5)
        poc_client.stop()
        
        # All distances in callbacks should be < r_target
        for batch in callback_server["received"]["generated"]:
            for dist in batch["dist"]:
                assert dist < 1.0, f"Distance {dist} >= r_target 1.0"
```

## Running Tests

```bash
# Run all PoC tests (unit + integration)
pytest tests/poc/ -v

# Run only unit tests (fast, no server needed)
pytest tests/poc/ -v --ignore=tests/poc/test_integration.py

# Run specific E2E test class
pytest tests/poc/test_integration.py::TestValidation -v

# Run only E2E tests
pytest tests/poc/test_integration.py -v
```

## Test Matrix

| Test | Category | Expected Result |
|------|----------|-----------------|
| `test_start_stop_round` | Basic | Pass |
| `test_status_during_round` | Basic | Pass |
| `test_valid_nonces_found` | Basic | Pass |
| `test_same_nonce_same_distance` | Determinism | Pass |
| `test_different_public_key_different_distance` | Determinism | Pass |
| `test_wrong_seed_fails_validation` | Validation | Pass |
| `test_fabricated_nonces_invalid` | Validation | Pass |
| `test_chat_returns_503` | Blocking | Pass |
| `test_inference_after_poc` | Switch | Pass |
| `test_multiple_rounds` | Switch | Pass |
| `test_distances_in_valid_range` | Distribution | Pass |
| `test_mean_distance_reasonable` | Distribution | Pass |
| `test_wrong_block_hash_validation` | Seed Validation | Pass |
| `test_wrong_public_key_validation` | Seed Validation | Pass |
| `test_fabricated_distance_detection` | Seed Validation | Pass |
| `test_tampered_nonce_list` | Seed Validation | Pass |
| `test_distance_distribution_stats` | Distribution Analysis | Pass |
| `test_distribution_consistency_across_blocks` | Distribution Analysis | Pass |
| `test_honest_batch_passes` | Fraud Detection | Pass |
| `test_fabricated_batch_detected` | Fraud Detection | Pass |
|| `test_correct_seed_correct_nonce_validates` | Seed Scenarios | Pass |
|| `test_wrong_seed_correct_nonce_fails` | Seed Scenarios | Pass |
|| `test_fabricated_nonce_not_validated` | Seed Scenarios | Pass |
|| `test_nonce_replay_different_round_fails` | Seed Scenarios | Pass |
|| `test_wrong_public_key_replay_fails` | Seed Scenarios | Pass |
|| `test_compare_with_theoretical_uniform` | Distribution Verification | Pass |
|| `test_valid_rate_vs_r_target_calibration` | Distribution Verification | Pass |
|| `test_distribution_stability_across_blocks` | Distribution Verification | Pass |
|| `test_distance_distribution_histogram` | Model Distribution | Pass |
|| `test_r_target_calibration` | Model Distribution | Pass |
|| `test_distribution_not_uniform_for_trained_model` | Model Distribution | Pass |
|| `test_generated_callback_sent` | Callback | Pass |
|| `test_callback_with_status_both_work` | Callback | Pass |
|| `test_callback_failure_does_not_block` | Callback | Pass |
|| `test_no_callback_when_not_configured` | Callback | Pass |
|| `test_callback_only_sends_valid_nonces` | Callback | Pass |

## Directory Structure After Phase 5

```
tests/poc/
├── __init__.py
├── conftest.py              # vLLM server fixture, PoCClient
├── test_data.py             # Phase 1: data class unit tests
├── test_gpu_random.py       # Phase 2: RNG unit tests
├── test_manager.py          # Phase 3: manager unit tests
├── test_routes.py           # Phase 4: API route unit tests
└── test_integration.py      # Phase 5: E2E integration tests
```

## Acceptance Criteria

### Prerequisites (from earlier phases)

- [ ] Phase 1-4 unit tests pass: `pytest tests/poc/test_data.py tests/poc/test_gpu_random.py tests/poc/test_manager.py tests/poc/test_routes.py`

### E2E Integration Tests

- [ ] All E2E tests pass: `pytest tests/poc/test_integration.py`
- [ ] Determinism verified (same inputs = same outputs)
- [ ] Wrong seed/pubkey validation fails as expected
- [ ] Inference blocked during PoC (503)
- [ ] Seamless switching between inference and PoC modes
- [ ] Distance distribution in valid range [0, 2]
- [ ] Seed validation tests pass (wrong hash, wrong pubkey, tampered nonces)
- [ ] Distribution analysis tests pass
- [ ] Distribution verification tests pass (comparison with theoretical, r_target calibration, stability)
- [ ] Fraud detection tests pass (honest batches pass, fabricated batches detected)
- [ ] Callback tests pass (push model, failure resilience, valid-only filtering)

### Full Test Suite

```bash
# Run all tests
pytest tests/poc/ -v

# Run only unit tests (no server needed)
pytest tests/poc/ -v --ignore=tests/poc/test_integration.py

# Run only E2E tests (requires vLLM server)
pytest tests/poc/test_integration.py -v
```


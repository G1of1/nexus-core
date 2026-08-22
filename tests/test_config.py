"""Tests for configuration validation and settings loading."""

import os
import pytest
from nexus.config import NexusSettings


def test_default_settings_values():
    """Test that default settings are reasonable when no env vars set."""
    import os
    # Temporarily unset environment variables that might override defaults
    env_backup = {}
    nexus_vars = [k for k in os.environ.keys() if k.startswith("NEXUS_")]
    for var in nexus_vars:
        env_backup[var] = os.environ.pop(var)
    
    try:
        settings = NexusSettings()
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 64
        assert settings.top_k == 5
        assert settings.rerank_top_k == 3
        assert settings.llm_temperature == 0.0
        assert settings.llm_max_tokens == 2048
    finally:
        # Restore environment variables
        for var, value in env_backup.items():
            os.environ[var] = value


def test_chunk_overlap_less_than_chunk_size():
    """Test that chunk overlap is less than chunk size."""
    settings = NexusSettings(chunk_size=100, chunk_overlap=50)
    assert settings.chunk_overlap < settings.chunk_size


def test_invalid_chunk_overlap_exceeds_size():
    """Test that chunk overlap cannot exceed chunk size."""
    with pytest.raises(ValueError):
        NexusSettings(chunk_size=100, chunk_overlap=150)


def test_invalid_negative_chunk_size():
    """Test that negative chunk size is rejected."""
    with pytest.raises(ValueError):
        NexusSettings(chunk_size=-1)


def test_invalid_negative_chunk_overlap():
    """Test that negative chunk overlap is rejected."""
    with pytest.raises(ValueError):
        NexusSettings(chunk_overlap=-1)


def test_top_k_greater_than_or_equal_rerank_top_k():
    """Test that top_k >= rerank_top_k."""
    settings = NexusSettings(top_k=10, rerank_top_k=5)
    assert settings.top_k >= settings.rerank_top_k


def test_invalid_rerank_top_k_exceeds_top_k():
    """Test that rerank_top_k cannot exceed top_k."""
    with pytest.raises(ValueError):
        NexusSettings(top_k=5, rerank_top_k=10)


def test_invalid_negative_top_k():
    """Test that negative top_k is rejected."""
    with pytest.raises(ValueError):
        NexusSettings(top_k=-1)


def test_invalid_negative_rerank_top_k():
    """Test that negative rerank_top_k is rejected."""
    with pytest.raises(ValueError):
        NexusSettings(rerank_top_k=-1)


def test_invalid_negative_vector_size():
    """Test that negative vector size is rejected."""
    with pytest.raises(ValueError):
        NexusSettings(vector_size=-1)


def test_invalid_temperature_out_of_range():
    """Test that temperature is in valid range [0, 2]."""
    with pytest.raises(ValueError):
        NexusSettings(llm_temperature=-1.0)
    
    with pytest.raises(ValueError):
        NexusSettings(llm_temperature=3.0)


def test_valid_temperature_range():
    """Test that valid temperatures are accepted."""
    settings = NexusSettings(llm_temperature=0.0)
    assert settings.llm_temperature == 0.0
    
    settings = NexusSettings(llm_temperature=1.5)
    assert settings.llm_temperature == 1.5
    
    settings = NexusSettings(llm_temperature=2.0)
    assert settings.llm_temperature == 2.0


def test_invalid_negative_max_tokens():
    """Test that negative max tokens is rejected."""
    with pytest.raises(ValueError):
        NexusSettings(llm_max_tokens=-1)


def test_invalid_zero_max_tokens():
    """Test that zero max tokens is rejected."""
    with pytest.raises(ValueError):
        NexusSettings(llm_max_tokens=0)


def test_environment_variable_override():
    """Test that environment variables override defaults."""
    os.environ["NEXUS_CHUNK_SIZE"] = "256"
    os.environ["NEXUS_TOP_K"] = "10"
    
    try:
        settings = NexusSettings()
        assert settings.chunk_size == 256
        assert settings.top_k == 10
    finally:
        # Clean up
        if "NEXUS_CHUNK_SIZE" in os.environ:
            del os.environ["NEXUS_CHUNK_SIZE"]
        if "NEXUS_TOP_K" in os.environ:
            del os.environ["NEXUS_TOP_K"]


def test_embedding_batch_size_positive():
    """Test that embedding batch size must be positive."""
    with pytest.raises(ValueError):
        NexusSettings(embedding_batch_size=0)
    
    with pytest.raises(ValueError):
        NexusSettings(embedding_batch_size=-5)


def test_valid_embedding_batch_size():
    """Test that valid embedding batch size is accepted."""
    settings = NexusSettings(embedding_batch_size=50)
    assert settings.embedding_batch_size == 50


def test_score_threshold_in_valid_range():
    """Test that score threshold is in valid range [0, 1]."""
    with pytest.raises(ValueError):
        NexusSettings(score_threshold=-0.1)
    
    with pytest.raises(ValueError):
        NexusSettings(score_threshold=1.5)


def test_valid_score_threshold():
    """Test that valid score thresholds are accepted."""
    settings = NexusSettings(score_threshold=0.0)
    assert settings.score_threshold == 0.0
    
    settings = NexusSettings(score_threshold=0.5)
    assert settings.score_threshold == 0.5
    
    settings = NexusSettings(score_threshold=1.0)
    assert settings.score_threshold == 1.0
    
    settings = NexusSettings(score_threshold=None)
    assert settings.score_threshold is None

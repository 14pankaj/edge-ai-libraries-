# Keep track of test paths
import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

# Base path for all test files
TEST_BASE_PATH = os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture(scope="session", autouse=True)
def setup_test_paths():
    """Create and clean up test directories"""
    # Create test directories if they don't exist
    os.makedirs(TEST_BASE_PATH, exist_ok=True)
    yield

@pytest.fixture
def test_model_path():
    """Fixture to provide model path within test directory"""
    path = os.path.join(TEST_BASE_PATH, "models")
    os.makedirs(path, exist_ok=True)
    return path

@pytest.fixture
def test_client():
    """TestClient fixture for FastAPI application"""
    return TestClient(app)

@pytest.fixture
def mock_hf_token():
    """Mock Hugging Face token"""
    return "mock_hf_token"

@pytest.fixture
def single_model_request(test_model_path):
    """Fixture for a valid single model request with test path"""
    return {
        "models": [
            {
                "name": "bert-base-uncased",
                "type": "llm",
                "is_ovms": True,
                "ovms_config": {
                    "precision": "int8",
                    "device": "CPU",
                    "cache_size": 10
                }
            }
        ],
        "parallel_downloads": False
    }

@pytest.fixture
def multi_model_request():
    """Fixture for a valid multi-model request"""
    return {
        "models": [
            {
                "name": "bert-base-uncased",
                "type": "llm",
                "is_ovms": True,
                "ovms_config": {
                    "precision": "int8",
                    "device": "CPU",
                    "cache_size": 10
                }
            },
            {
                "name": "sentence-transformers/all-mpnet-base-v2",
                "type": "embeddings",
                "is_ovms": True,
                "ovms_config": {
                    "precision": "fp16",
                    "device": "GPU",
                    "cache_size": 20
                }
            }
        ],
        "parallel_downloads": True
    }

@pytest.fixture
def invalid_model_requests():
    """Fixture for various invalid request scenarios"""
    return {
        "empty_models": {"models": []},
        "missing_name": {
            "models": [{"type": "llm", "is_ovms": True}]
        },
        "invalid_type": {
            "models": [{
                "name": "bert-base-uncased",
                "type": "invalid",
                "is_ovms": True
            }]
        },
        "invalid_ovms_config": {
            "models": [{
                "name": "bert-base-uncased",
                "type": "llm",
                "is_ovms": True,
                "ovms_config": {
                    "precision": "invalid",
                    "device": "invalid",
                    "cache_size": -1
                }
            }]
        },
        "invalid_revision": {
            "models": [{
                "name": "bert-base-uncased",
                "revision": 123,  # Should be string
                "is_ovms": True
            }]
        }
    }

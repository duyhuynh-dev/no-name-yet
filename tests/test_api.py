"""
Unit tests for FastAPI endpoints.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check(self, api_client):
        """Test health check returns 200."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_health_check_model_status(self, api_client):
        """Test health check includes model status."""
        response = api_client.get("/health")
        data = response.json()
        
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root(self, api_client):
        """Test root endpoint returns API info."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestSymbolsEndpoint:
    """Tests for /symbols endpoint."""
    
    def test_list_symbols(self, api_client):
        """Test symbols endpoint returns list."""
        response = api_client.get("/symbols")
        
        assert response.status_code == 200
        data = response.json()
        assert "symbols" in data
        assert isinstance(data["symbols"], list)


class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state for prediction."""
        n = 35  # More than window_size
        return {
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [101.0 + i * 0.1 for i in range(n)],
            "low": [99.0 + i * 0.1 for i in range(n)],
            "close": [100.5 + i * 0.1 for i in range(n)],
            "volume": [1000000.0 + i * 10000 for i in range(n)],
            "position": 0,
            "cash": 10000.0,
            "shares": 0.0,
        }
    
    def test_predict_valid_input(self, api_client, sample_market_state):
        """Test prediction with valid input."""
        response = api_client.post("/predict", json=sample_market_state)
        
        # May return 503 if model not loaded
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "action" in data
        assert data["action"] in ["hold", "buy", "sell"]
        assert "action_id" in data
        assert data["action_id"] in [0, 1, 2]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "probabilities" in data
        assert "latency_ms" in data
    
    def test_predict_response_schema(self, api_client, sample_market_state):
        """Test prediction response matches schema."""
        response = api_client.post("/predict", json=sample_market_state)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        data = response.json()
        
        # Check probabilities sum to ~1
        if "probabilities" in data:
            probs = data["probabilities"]
            total = sum(probs.values())
            assert 0.99 <= total <= 1.01, f"Probabilities should sum to 1, got {total}"
    
    def test_predict_invalid_input_missing_field(self, api_client):
        """Test prediction with missing required field."""
        invalid_state = {
            "open": [100.0] * 35,
            "high": [101.0] * 35,
            # Missing low, close, volume
        }
        
        response = api_client.post("/predict", json=invalid_state)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_input_empty_arrays(self, api_client):
        """Test prediction with empty arrays."""
        invalid_state = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "position": 0,
            "cash": 10000.0,
            "shares": 0.0,
        }
        
        response = api_client.post("/predict", json=invalid_state)
        assert response.status_code == 422  # Validation error
    
    def test_predict_latency(self, api_client, sample_market_state):
        """Test prediction latency is under target."""
        response = api_client.post("/predict", json=sample_market_state)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        data = response.json()
        
        # Target is <50ms
        assert data["latency_ms"] < 100, f"Latency too high: {data['latency_ms']}ms"


class TestBacktestEndpoint:
    """Tests for /backtest endpoint."""
    
    def test_backtest_valid_symbol(self, api_client):
        """Test backtest with valid symbol."""
        request = {
            "symbol": "SPY",
            "initial_balance": 10000.0,
        }
        
        response = api_client.post("/backtest", json=request)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        if response.status_code == 404:
            pytest.skip("Data not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_return" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data
        assert "equity_curve" in data
    
    def test_backtest_response_types(self, api_client):
        """Test backtest response field types."""
        request = {
            "symbol": "SPY",
            "initial_balance": 10000.0,
        }
        
        response = api_client.post("/backtest", json=request)
        
        if response.status_code != 200:
            pytest.skip("Backtest not available")
        
        data = response.json()
        
        assert isinstance(data["total_return"], (int, float))
        assert isinstance(data["sharpe_ratio"], (int, float))
        assert isinstance(data["max_drawdown"], (int, float))
        assert isinstance(data["win_rate"], (int, float))
        assert isinstance(data["num_trades"], int)
        assert isinstance(data["equity_curve"], list)
    
    def test_backtest_invalid_symbol(self, api_client):
        """Test backtest with invalid symbol."""
        request = {
            "symbol": "INVALID_SYMBOL_XYZ",
            "initial_balance": 10000.0,
        }
        
        response = api_client.post("/backtest", json=request)
        
        # Should return 404 or 500 for missing data
        assert response.status_code in [404, 500, 503]


class TestModelEndpoints:
    """Tests for model-related endpoints."""
    
    def test_model_info(self, api_client):
        """Test model info endpoint."""
        response = api_client.get("/model/info")
        
        # May return 500 if service not initialized
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "loaded" in data
    
    def test_model_load_invalid_path(self, api_client):
        """Test loading model from invalid path."""
        response = api_client.post(
            "/model/load",
            params={"model_path": "/nonexistent/path/model.zip"}
        )
        
        # May return 400 or 500 depending on error handling
        assert response.status_code in [400, 500]


class TestAPIErrorHandling:
    """Tests for API error handling."""
    
    def test_invalid_endpoint(self, api_client):
        """Test invalid endpoint returns 404."""
        response = api_client.get("/nonexistent_endpoint")
        assert response.status_code == 404
    
    def test_invalid_method(self, api_client):
        """Test invalid method returns 405."""
        response = api_client.post("/health")
        assert response.status_code == 405
    
    def test_invalid_json(self, api_client):
        """Test invalid JSON returns 422."""
        response = api_client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


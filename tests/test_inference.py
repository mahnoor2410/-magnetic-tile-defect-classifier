"""
Tests for the inference pipeline.

Run (from the project root):

    pytest tests/ -v

These tests are self-contained: they build a tiny randomly-initialized model
and checkpoint on the fly rather than depending on a real trained checkpoint
being present, so they run in CI / any fresh clone of the repo without
requiring the dataset or a completed training run.
"""

import io
import os
import sys

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.resnet_finetune import build_resnet_model
from src.inference.predictor import DefectPredictor

TEST_CLASSES = ["Blowhole", "Break", "Crack", "Fray", "Uneven", "Free"]


@pytest.fixture(scope="module")
def dummy_checkpoint(tmp_path_factory):
    """Create a randomly-initialized ResNet18 checkpoint with valid metadata,
    saved to a temporary directory, purely for pipeline testing."""
    checkpoint_dir = tmp_path_factory.mktemp("checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "test_resnet.pth")

    model = build_resnet_model(num_classes=len(TEST_CLASSES), pretrained=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": TEST_CLASSES,
        "model_type": "resnet",
        "image_size": 224,
    }, checkpoint_path)

    return checkpoint_path


@pytest.fixture(scope="module")
def predictor(dummy_checkpoint):
    return DefectPredictor(
        checkpoint_path=dummy_checkpoint,
        model_type="resnet",
        device="cpu",
        image_size=224,
    )


def _make_dummy_image_bytes(size=(224, 224), color=(120, 120, 120)) -> bytes:
    """Generate a simple in-memory RGB JPEG image for testing."""
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class TestDefectPredictor:
    def test_predictor_loads_successfully(self, predictor):
        assert predictor.classes == TEST_CLASSES
        assert predictor.model_type == "resnet"

    def test_preprocess_returns_correct_shape(self, predictor):
        image_bytes = _make_dummy_image_bytes()
        tensor = predictor.preprocess(image_bytes)
        assert tensor.shape == (1, 3, 224, 224)

    def test_predict_returns_valid_structure(self, predictor):
        image_bytes = _make_dummy_image_bytes()
        result = predictor.predict(image_bytes)

        assert "predicted_class" in result
        assert "confidence" in result
        assert "all_probabilities" in result
        assert result["predicted_class"] in TEST_CLASSES
        assert 0.0 <= result["confidence"] <= 1.0
        assert set(result["all_probabilities"].keys()) == set(TEST_CLASSES)

    def test_probabilities_sum_to_one(self, predictor):
        image_bytes = _make_dummy_image_bytes()
        result = predictor.predict(image_bytes)
        total = sum(result["all_probabilities"].values())
        assert abs(total - 1.0) < 1e-2

    def test_predict_rejects_corrupt_image(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict(b"not a real image")

    def test_missing_checkpoint_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DefectPredictor(checkpoint_path="nonexistent/path.pth", model_type="resnet")


class TestAPI:
    """
    Tests the FastAPI app directly against a dummy checkpoint by monkeypatching
    the module-level predictor, so these tests do not require a real trained
    model checkpoint to exist on disk.
    """

    @pytest.fixture(scope="class")
    def client(self, dummy_checkpoint):
        # Import here (not at module scope) so we can patch the global
        # predictor before any route is exercised.
        from src.inference import api as api_module

        api_module.predictor = DefectPredictor(
            checkpoint_path=dummy_checkpoint, model_type="resnet", device="cpu",
        )
        api_module.model_load_error = None

        return TestClient(api_module.app)

    def test_health_endpoint_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True

    def test_classes_endpoint(self, client):
        response = client.get("/classes")
        assert response.status_code == 200
        assert response.json()["classes"] == TEST_CLASSES

    def test_predict_endpoint_valid_image(self, client):
        image_bytes = _make_dummy_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_class"] in TEST_CLASSES
        assert "confidence" in body

    def test_predict_endpoint_rejects_bad_content_type(self, client):
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_endpoint_rejects_empty_file(self, client):
        response = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_predict_endpoint_rejects_corrupt_image(self, client):
        response = client.post(
            "/predict",
            files={"file": ("corrupt.jpg", b"not a real image", "image/jpeg")},
        )
        assert response.status_code == 400

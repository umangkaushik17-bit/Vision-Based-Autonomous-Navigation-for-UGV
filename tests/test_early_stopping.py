"""Unit tests for EarlyStopping mechanism and training configuration."""

import pytest
import yaml
from pathlib import Path
from ai.training.early_stopping import EarlyStopping


def test_early_stopping_invalid_params():
    """Test parameter validation on initialization."""
    with pytest.raises(ValueError, match="patience must be a positive integer"):
        EarlyStopping(patience=0)

    with pytest.raises(ValueError, match="patience must be a positive integer"):
        EarlyStopping(patience=-3)

    with pytest.raises(ValueError, match="min_delta must be non-negative"):
        EarlyStopping(min_delta=-0.01)

    with pytest.raises(ValueError, match="mode must be either 'min' or 'max'"):
        EarlyStopping(mode="invalid_mode")


def test_early_stopping_max_mode():
    """Test early stopping in 'max' mode (higher is better, e.g. mIoU)."""
    stopper = EarlyStopping(patience=3, min_delta=0.01, mode="max", monitor="val_mIoU")

    assert stopper.counter == 0
    assert stopper.best_score is None
    assert stopper.early_stop is False

    # Epoch 1: Initial metric
    assert stopper.step(0.50, epoch=1) is False
    assert stopper.best_score == 0.50
    assert stopper.best_epoch == 1
    assert stopper.is_improved is True
    assert stopper.counter == 0

    # Epoch 2: Improvement exceeding min_delta (0.50 -> 0.52)
    assert stopper.step(0.52, epoch=2) is False
    assert stopper.best_score == 0.52
    assert stopper.best_epoch == 2
    assert stopper.is_improved is True
    assert stopper.counter == 0

    # Epoch 3: No improvement (0.52 -> 0.525, delta 0.005 < min_delta 0.01)
    assert stopper.step(0.525, epoch=3) is False
    assert stopper.is_improved is False
    assert stopper.counter == 1
    assert stopper.best_score == 0.52
    assert stopper.best_epoch == 2

    # Epoch 4: Decrease
    assert stopper.step(0.51, epoch=4) is False
    assert stopper.counter == 2
    assert stopper.is_improved is False

    # Epoch 5: 3rd consecutive epoch without improvement -> Early stop triggered!
    assert stopper.step(0.515, epoch=5) is True
    assert stopper.counter == 3
    assert stopper.early_stop is True
    assert stopper.best_epoch == 2


def test_early_stopping_min_mode():
    """Test early stopping in 'min' mode (lower is better, e.g. loss)."""
    stopper = EarlyStopping(patience=2, min_delta=0.05, mode="min", monitor="loss")

    # Epoch 1
    assert stopper.step(1.0, epoch=1) is False
    assert stopper.best_score == 1.0
    assert stopper.is_improved is True

    # Epoch 2: Improvement (1.0 -> 0.8)
    assert stopper.step(0.8, epoch=2) is False
    assert stopper.best_score == 0.8
    assert stopper.counter == 0

    # Epoch 3: Insufficient improvement (0.8 -> 0.78, delta 0.02 < 0.05)
    assert stopper.step(0.78, epoch=3) is False
    assert stopper.counter == 1

    # Epoch 4: Increase in loss -> Early stop triggered!
    assert stopper.step(0.85, epoch=4) is True
    assert stopper.counter == 2
    assert stopper.early_stop is True


def test_early_stopping_state_dict_resumption():
    """Test checkpoint serialization and restoration."""
    stopper1 = EarlyStopping(patience=5, min_delta=0.001, mode="max", monitor="val_mIoU")
    stopper1.step(0.60, epoch=1)
    stopper1.step(0.605, epoch=2) # counter = 0, best = 0.605
    stopper1.step(0.600, epoch=3) # counter = 1
    stopper1.step(0.590, epoch=4) # counter = 2

    state = stopper1.state_dict()
    assert state["counter"] == 2
    assert state["best_score"] == 0.605
    assert state["best_epoch"] == 2

    stopper2 = EarlyStopping(patience=5, min_delta=0.001, mode="max", monitor="val_mIoU")
    stopper2.load_state_dict(state)

    assert stopper2.counter == 2
    assert stopper2.best_score == 0.605
    assert stopper2.best_epoch == 2
    assert stopper2.early_stop is False


def test_yaml_config_epochs_and_early_stopping():
    """Verify that configs/traversability_training.yaml specifies 50 epochs and early stopping."""
    config_path = Path("configs/traversability_training.yaml")
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Verify 50 epochs
    assert cfg["training"]["epochs"] == 50
    assert cfg["scheduler"]["T_max"] == 50

    # Verify early stopping configuration
    early_stop_cfg = cfg["training"].get("early_stopping", {})
    assert early_stop_cfg.get("enabled") is True
    assert early_stop_cfg.get("min_epochs") == 25
    assert early_stop_cfg.get("patience") >= 1
    assert early_stop_cfg.get("min_delta") >= 0.0
    assert early_stop_cfg.get("monitor") == "val_mIoU"
    assert early_stop_cfg.get("mode") == "max"


def test_early_stopping_min_epochs():
    """Verify that early stopping never triggers before min_epochs is reached."""
    stopper = EarlyStopping(patience=3, min_delta=0.01, mode="max", min_epochs=25)

    # Epochs 1 to 24: stagnant metric with no improvement
    for ep in range(1, 25):
        stop = stopper.step(0.50, epoch=ep)
        assert stop is False, f"Early stop triggered prematurely at epoch {ep} < min_epochs (25)"
        assert stopper.counter == 0, f"Counter should remain 0 during mandatory warm-up (epoch {ep})"

    # Epoch 25: still no improvement
    assert stopper.step(0.50, epoch=25) is False
    assert stopper.counter == 1

    # Epoch 26: no improvement
    assert stopper.step(0.50, epoch=26) is False
    assert stopper.counter == 2

    # Epoch 27: 3rd epoch without improvement after min_epochs -> Early stop triggered!
    assert stopper.step(0.50, epoch=27) is True
    assert stopper.counter == 3
    assert stopper.early_stop is True

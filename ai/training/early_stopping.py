"""Early Stopping mechanism for deep learning training loops.

Monitors validation or training metrics and stops training when no significant
improvement is observed for a specified number of consecutive epochs (patience).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class EarlyStopping:
    """Early stops training when a monitored metric stops improving.

    Attributes:
        patience: Number of epochs with no improvement after which training stops.
        min_delta: Minimum change in monitored metric to qualify as an improvement.
        mode: One of 'max' (higher is better, e.g. mIoU) or 'min' (lower is better, e.g. loss).
        monitor: Name of the metric being monitored.
        verbose: Whether to format detailed status messages.
        counter: Number of consecutive epochs without improvement.
        best_score: Best metric score observed so far.
        best_epoch: Epoch index (1-based) where the best score was recorded.
        early_stop: True if early stopping criteria has been triggered.
        is_improved: True if the most recent step produced an improvement.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "max",
        monitor: str = "val_mIoU",
        verbose: bool = True,
        min_epochs: int = 0
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be a positive integer, got {patience}")
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        if min_epochs < 0:
            raise ValueError(f"min_epochs must be non-negative, got {min_epochs}")

        mode_clean = mode.strip().lower()
        if mode_clean not in ("min", "max"):
            raise ValueError(f"mode must be either 'min' or 'max', got {mode!r}")

        self.patience: int = patience
        self.min_delta: float = float(min_delta)
        self.mode: str = mode_clean
        self.monitor: str = monitor
        self.verbose: bool = verbose
        self.min_epochs: int = int(min_epochs)

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.early_stop: bool = False
        self.is_improved: bool = False

    def step(self, current: float, epoch: int = 0) -> bool:
        """Evaluate the metric for the current epoch and update internal state.

        Args:
            current: Current epoch's metric value.
            epoch: Current epoch number (typically 1-indexed).

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        current_val = float(current)

        if self.best_score is None:
            self.best_score = current_val
            self.best_epoch = epoch
            self.is_improved = True
            self.counter = 0
            return False

        if self.mode == "max":
            improved = (current_val - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - current_val) > self.min_delta

        if improved:
            self.best_score = current_val
            self.best_epoch = epoch
            self.counter = 0
            self.is_improved = True
        else:
            self.is_improved = False
            # Only accumulate patience and trigger early stopping once epoch reaches min_epochs
            if epoch >= self.min_epochs:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0

        return self.early_stop

    def state_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "mode": self.mode,
            "monitor": self.monitor,
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "early_stop": self.early_stop,
            "is_improved": self.is_improved
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from a checkpoint dictionary."""
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.min_epochs = state.get("min_epochs", getattr(self, "min_epochs", 25))
        self.mode = state.get("mode", self.mode)
        self.monitor = state.get("monitor", self.monitor)
        self.counter = state.get("counter", 0)
        self.best_score = state.get("best_score", None)
        self.best_epoch = state.get("best_epoch", 0)
        self.early_stop = state.get("early_stop", False)
        self.is_improved = state.get("is_improved", False)

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(monitor='{self.monitor}', mode='{self.mode}', "
            f"min_epochs={self.min_epochs}, patience={self.patience}, min_delta={self.min_delta}, "
            f"counter={self.counter}/{self.patience}, best={self.best_score})"
        )

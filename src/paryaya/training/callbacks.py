"""
paryaya.training.callbacks — Training callbacks: early stopping + checkpoint management.

EarlyStopping: tracks best WER, triggers after `patience` non-improving epochs.
CheckpointManager: saves up to keep_top_k checkpoints by WER, deletes the worst.

Both expose state_dict() / load_state_dict() so they survive --resume restarts.

Usage:
    es  = EarlyStopping(patience=15)
    mgr = CheckpointManager("checkpoints/", keep_top_k=3)

    for epoch in ...:
        wer = valid(...)
        mgr.save_checkpoint(model, epoch, optimizer, wer)
        if es.update(wer):
            break
"""
from pathlib import Path


class EarlyStopping:
    """Stop training when WER has not improved by min_delta for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001) -> None:
        self.patience         = patience
        self.min_delta        = min_delta
        self.best_wer         = float("inf")
        self.steps_since_best = 0

    def update(self, wer: float) -> bool:
        """Register a new WER. Returns True if training should stop."""
        if wer < self.best_wer - self.min_delta:
            self.best_wer         = wer
            self.steps_since_best = 0
            return False
        self.steps_since_best += 1
        return self.steps_since_best >= self.patience

    def state_dict(self) -> dict:
        return {
            "best_wer":         self.best_wer,
            "steps_since_best": self.steps_since_best,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_wer         = state["best_wer"]
        self.steps_since_best = state["steps_since_best"]


class CheckpointManager:
    """Keep only the top-k checkpoints by WER; delete the rest automatically."""

    def __init__(self, out_dir: str | Path, keep_top_k: int = 3) -> None:
        self.out_dir    = Path(out_dir)
        self.keep_top_k = keep_top_k
        # List of (wer, path) sorted best-first (ascending WER)
        self._ckpts: list[tuple[float, Path]] = []

    def save_checkpoint(
        self,
        model,
        epoch: int,
        optimizer,
        wer: float,
    ) -> Path:
        """Persist checkpoint and evict the worst if over keep_top_k."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"model_ep{epoch:03d}_wer{wer:.4f}.pt"
        model.save_checkpoint(str(path), epoch, optimizer, wer)

        self._ckpts.append((wer, path))
        self._ckpts.sort(key=lambda t: t[0])   # best (lowest WER) first

        while len(self._ckpts) > self.keep_top_k:
            _, worst = self._ckpts.pop()        # pop highest WER
            if worst.exists():
                worst.unlink()

        return path

    @property
    def best_wer(self) -> float:
        return self._ckpts[0][0] if self._ckpts else float("inf")

    @property
    def best_path(self) -> Path | None:
        return self._ckpts[0][1] if self._ckpts else None

    def state_dict(self) -> dict:
        return {"checkpoints": [(wer, str(p)) for wer, p in self._ckpts]}

    def load_state_dict(self, state: dict) -> None:
        self._ckpts = [(wer, Path(p)) for wer, p in state.get("checkpoints", [])]

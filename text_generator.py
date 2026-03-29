#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import pickle
import re
import shlex
import shutil
import sys
import unicodedata
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

CHECKPOINTS_DIR = Path(__file__).with_name("checkpoints")
DEFAULT_DATA_PATH = Path(__file__).with_name("shakespeare.txt")
LEGACY_CHECKPOINT_PATH = CHECKPOINTS_DIR / "latest_model.npz"
MANAGED_SESSIONS_DIR = CHECKPOINTS_DIR / "sessions"
DEFAULT_STARTER_CHECKPOINT_PATH = CHECKPOINTS_DIR / "starter_model.npz"
SESSION_CHECKPOINT_FILENAME = "latest_model.npz"
SESSION_LOCK_FILENAME = "session.lock"
PAD_TOKEN = "\0"
PROMPT_TRANSLATION_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\t": " ",
    }
)
DEFAULT_SHELL_SAMPLE_LENGTH = 300
DEFAULT_SHELL_TRAIN_STEPS = 500
DEFAULT_SESSION_NAME = "default"
SESSION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class SessionError(ValueError):
    pass


class SessionLockError(RuntimeError):
    pass


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def corpus_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_prompt(prompt: str) -> str:
    return unicodedata.normalize("NFKC", prompt).translate(PROMPT_TRANSLATION_TABLE)


def sanitize_prompt(prompt: str, corpus: Corpus) -> tuple[str, list[str]]:
    normalized = normalize_prompt(prompt)
    notes: list[str] = []
    if normalized != prompt:
        notes.append("Normalized punctuation in the prompt.")

    cleaned_chars: list[str] = []
    dropped_chars: list[str] = []
    seen_dropped: set[str] = set()
    for char in normalized:
        if char in corpus.stoi:
            cleaned_chars.append(char)
            continue
        if char not in seen_dropped:
            seen_dropped.add(char)
            dropped_chars.append(char)

    if dropped_chars:
        preview = "".join(dropped_chars[:10])
        notes.append(f"Removed unsupported characters from the prompt: {preview!r}")
    if prompt and not cleaned_chars:
        notes.append("The prompt ended up empty, so generation started from a blank prompt.")

    return "".join(cleaned_chars), notes


def resolve_checkpoint_argument(checkpoint: str) -> Path:
    return Path(checkpoint).expanduser().resolve()


def resolve_starter_checkpoint_path() -> Path | None:
    safe_path = DEFAULT_STARTER_CHECKPOINT_PATH.expanduser().resolve()
    if safe_path.exists():
        return safe_path
    return None


def explicit_checkpoint_lock_path(checkpoint_path: Path) -> Path:
    resolved = checkpoint_path.expanduser().resolve()
    return resolved.parent / f"{resolved.name}.lock"


@dataclass(frozen=True)
class RunTarget:
    checkpoint_path: Path
    lock_path: Path | None
    session_name: str | None
    managed: bool

    @property
    def label(self) -> str:
        if self.managed:
            return f"Session '{self.session_name}'"
        return f"Checkpoint {self.checkpoint_path}"


@dataclass
class LockHandle:
    path: Path
    label: str
    handle: Any

    def release(self) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None

    def __enter__(self) -> "LockHandle":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


def acquire_exclusive_lock(lock_path: Path, label: str) -> LockHandle:
    resolved = lock_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    handle = resolved.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SessionLockError(
            f"{label} is already in use. Choose another session or close the other process."
        ) from exc
    return LockHandle(path=resolved, label=label, handle=handle)


def acquire_target_lock(target: RunTarget) -> contextlib.AbstractContextManager[LockHandle | None]:
    if target.lock_path is None:
        return contextlib.nullcontext()
    return acquire_exclusive_lock(target.lock_path, target.label)


class SessionManager:
    def __init__(
        self,
        *,
        sessions_root: Path | None = None,
        legacy_checkpoint_path: Path | None = None,
    ) -> None:
        root = MANAGED_SESSIONS_DIR if sessions_root is None else sessions_root
        legacy = LEGACY_CHECKPOINT_PATH if legacy_checkpoint_path is None else legacy_checkpoint_path
        self.sessions_root = root.expanduser().resolve()
        self.legacy_checkpoint_path = legacy.expanduser().resolve()

    def validate_name(self, name: str) -> str:
        cleaned = name.strip()
        if not cleaned:
            raise SessionError("Session names cannot be empty.")
        if SESSION_NAME_PATTERN.fullmatch(cleaned) is None:
            raise SessionError(
                "Session names may use only ASCII letters, digits, '.', '_' and '-'."
            )
        return cleaned

    def session_dir(self, name: str) -> Path:
        validated = self.validate_name(name)
        return self.sessions_root / validated

    def checkpoint_path(self, name: str) -> Path:
        return self.session_dir(name) / SESSION_CHECKPOINT_FILENAME

    def lock_path(self, name: str) -> Path:
        return self.session_dir(name) / SESSION_LOCK_FILENAME

    def ensure_session_dir(self, name: str) -> Path:
        directory = self.session_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def target_for_session(self, name: str) -> RunTarget:
        validated = self.validate_name(name)
        self.ensure_session_dir(validated)
        return RunTarget(
            checkpoint_path=self.checkpoint_path(validated),
            lock_path=self.lock_path(validated),
            session_name=validated,
            managed=True,
        )

    def target_for_checkpoint(self, checkpoint_path: Path) -> RunTarget:
        resolved = checkpoint_path.expanduser().resolve()
        return RunTarget(
            checkpoint_path=resolved,
            lock_path=explicit_checkpoint_lock_path(resolved),
            session_name=None,
            managed=False,
        )

    def list_sessions(self) -> list[str]:
        if not self.sessions_root.exists():
            return []
        names = [
            path.name
            for path in self.sessions_root.iterdir()
            if path.is_dir() and SESSION_NAME_PATTERN.fullmatch(path.name)
        ]
        return sorted(names)

    def is_locked(self, name: str) -> bool:
        validated = self.validate_name(name)
        if not self.session_dir(validated).exists():
            return False
        try:
            lock = acquire_exclusive_lock(self.lock_path(validated), f"Session '{validated}'")
        except SessionLockError:
            return True
        lock.release()
        return False

    def migrate_legacy_checkpoint(self) -> list[str]:
        messages: list[str] = []
        legacy_path = self.legacy_checkpoint_path
        default_checkpoint_path = self.checkpoint_path(DEFAULT_SESSION_NAME)

        if not legacy_path.exists():
            return messages

        if default_checkpoint_path.exists():
            messages.append(
                "Warning: a legacy checkpoint and session 'default' both exist. "
                "Using the session copy and leaving the legacy checkpoint untouched."
            )
            return messages

        default_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.replace(default_checkpoint_path)
        messages.append(
            "Migrated legacy saved progress into session 'default'."
        )
        return messages

    def choose_session_interactively(self) -> str:
        while True:
            session_names = self.list_sessions()
            print("Managed sessions:")
            if session_names:
                for index, name in enumerate(session_names, start=1):
                    status = "locked" if self.is_locked(name) else "available"
                    print(f"  {index}. {name} ({status})")
            else:
                print("  (none yet)")

            raw = input("Choose a session number or enter a new session name: ").strip()
            if not raw:
                print("Please enter a session number or a new session name.")
                continue

            if raw.isdigit():
                index = int(raw)
                if 1 <= index <= len(session_names):
                    chosen = session_names[index - 1]
                else:
                    chosen = raw
            else:
                chosen = raw

            try:
                validated = self.validate_name(chosen)
            except SessionError as exc:
                print(exc)
                continue

            if self.is_locked(validated):
                print(
                    f"Session '{validated}' is already in use. "
                    "Choose another session or close the other process."
                )
                continue

            self.ensure_session_dir(validated)
            return validated

    def delete_session(self, name: str, *, current_session: str | None = None) -> str:
        validated = self.validate_name(name)
        if current_session == validated:
            raise ValueError("Refusing to delete the current session.")

        session_dir = self.session_dir(validated)
        if not session_dir.exists():
            return f"No session named '{validated}'."

        with acquire_exclusive_lock(self.lock_path(validated), f"Session '{validated}'"):
            shutil.rmtree(session_dir)
        return f"Deleted session '{validated}'."


@dataclass
class Corpus:
    text: str
    context_size: int
    pad_id: int = 0
    chars: list[str] = field(init=False)
    stoi: dict[str, int] = field(init=False)
    itos: dict[int, str] = field(init=False)
    encoded: np.ndarray = field(init=False)
    train_encoded: np.ndarray = field(init=False)
    val_encoded: np.ndarray = field(init=False)
    train_padded: np.ndarray = field(init=False)
    val_padded: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.context_size <= 0:
            raise ValueError("context_size must be a positive integer.")

        unique_chars = sorted(set(self.text))
        if PAD_TOKEN in unique_chars:
            raise ValueError("The training text already contains the reserved padding token.")
        if len(self.text) < 2:
            raise ValueError("The training text must contain at least two characters.")

        self.chars = [PAD_TOKEN] + unique_chars
        self.stoi = {ch: idx for idx, ch in enumerate(self.chars)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}

        self.encoded = np.fromiter((self.stoi[ch] for ch in self.text), dtype=np.int32)
        split_at = max(1, int(len(self.encoded) * 0.9))
        split_at = min(split_at, len(self.encoded) - 1)

        self.train_encoded = self.encoded[:split_at]
        self.val_encoded = self.encoded[split_at:]
        self.train_padded = self._pad(self.train_encoded)
        self.val_padded = self._pad(self.val_encoded)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def _pad(self, data: np.ndarray) -> np.ndarray:
        pad = np.full(self.context_size, self.pad_id, dtype=np.int32)
        return np.concatenate((pad, data.astype(np.int32, copy=False)))

    def encode_prompt(self, prompt: str) -> list[int]:
        unknown = sorted({ch for ch in prompt if ch not in self.stoi})
        if unknown:
            unknown_preview = "".join(unknown[:10])
            raise ValueError(
                f"Prompt contains characters outside the training vocabulary: {unknown_preview!r}"
            )
        return [self.stoi[ch] for ch in prompt]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids if token_id != self.pad_id)


class CharMLP:
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embed_dim: int = 24,
        hidden_dim: int = 192,
        seed: int = 42,
    ) -> None:
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1.")
        if context_size <= 0:
            raise ValueError("context_size must be a positive integer.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)

        scale_embed = 0.05
        scale_hidden = (2.0 / (context_size * embed_dim)) ** 0.5
        scale_out = (2.0 / hidden_dim) ** 0.5

        self.params: dict[str, np.ndarray] = {
            "E": self.rng.normal(0.0, scale_embed, size=(vocab_size, embed_dim)).astype(np.float32),
            "W1": self.rng.normal(
                0.0, scale_hidden, size=(context_size * embed_dim, hidden_dim)
            ).astype(np.float32),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "W2": self.rng.normal(0.0, scale_out, size=(hidden_dim, vocab_size)).astype(
                np.float32
            ),
            "b2": np.zeros(vocab_size, dtype=np.float32),
        }

        self.opt_state: dict[str, Any] = {
            "step": 0,
            "m": {name: np.zeros_like(value) for name, value in self.params.items()},
            "v": {name: np.zeros_like(value) for name, value in self.params.items()},
        }
        self._scratch = {name: np.empty_like(value) for name, value in self.params.items()}
        self._denom_scratch = {name: np.empty_like(value) for name, value in self.params.items()}

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        embedding = self.params["E"][x]
        flat = embedding.reshape(x.shape[0], self.context_size * self.embed_dim)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            hidden_linear = flat @ self.params["W1"] + self.params["b1"]
        hidden = np.tanh(hidden_linear)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            logits = hidden @ self.params["W2"] + self.params["b2"]
        cache = {"x": x, "flat": flat, "hidden": hidden}
        return logits, cache

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        logits, _ = self.forward(x)
        probs = self._softmax(logits)
        return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        logits, cache = self.forward(x)
        probs = self._softmax(logits)
        batch_size = len(y)
        loss = float(-np.log(probs[np.arange(batch_size), y] + 1e-12).mean())

        dlogits = probs
        dlogits[np.arange(batch_size), y] -= 1.0
        dlogits /= batch_size

        grads: dict[str, np.ndarray] = {}
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            grads["W2"] = cache["hidden"].T @ dlogits
        grads["b2"] = dlogits.sum(axis=0)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            dhidden = dlogits @ self.params["W2"].T
        dhidden_linear = dhidden * (1.0 - cache["hidden"] ** 2)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            grads["W1"] = cache["flat"].T @ dhidden_linear
        grads["b1"] = dhidden_linear.sum(axis=0)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            dflat = dhidden_linear @ self.params["W1"].T
        dembedding = dflat.reshape(batch_size, self.context_size, self.embed_dim)
        grads["E"] = self._accumulate_embedding_grads(cache["x"], dembedding)

        if not np.isfinite(loss):
            raise FloatingPointError("Loss became non-finite during training.")
        for name, grad in grads.items():
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError(f"Gradient {name} became non-finite during training.")

        return loss, grads

    def _accumulate_embedding_grads(
        self,
        token_ids: np.ndarray,
        embedding_grads: np.ndarray,
    ) -> np.ndarray:
        flat_token_ids = token_ids.reshape(-1)
        flat_embedding_grads = embedding_grads.reshape(-1, self.embed_dim)
        grad = np.zeros_like(self.params["E"])
        if flat_token_ids.size == 0:
            return grad

        order = np.argsort(flat_token_ids, kind="stable")
        sorted_token_ids = flat_token_ids[order]
        unique_token_ids, starts = np.unique(sorted_token_ids, return_index=True)
        grad[unique_token_ids] = np.add.reduceat(flat_embedding_grads[order], starts, axis=0)
        return grad

    def apply_grads(
        self,
        grads: dict[str, np.ndarray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float = 1.0,
    ) -> float:
        total_norm = float(np.sqrt(sum(np.sum(grad**2) for grad in grads.values())))
        clip_scale = 1.0
        if total_norm > grad_clip:
            clip_scale = grad_clip / (total_norm + 1e-12)

        self.opt_state["step"] += 1
        step = self.opt_state["step"]
        m_state = self.opt_state["m"]
        v_state = self.opt_state["v"]
        beta1_correction = 1.0 - beta1**step
        beta2_correction = 1.0 - beta2**step
        one_minus_beta1 = 1.0 - beta1
        one_minus_beta2 = 1.0 - beta2

        for name, param in self.params.items():
            grad = grads[name]
            m = m_state[name]
            v = v_state[name]
            scratch = self._scratch[name]
            denom = self._denom_scratch[name]

            if clip_scale != 1.0:
                np.multiply(grad, clip_scale, out=scratch)
                grad = scratch

            m *= beta1
            np.multiply(grad, one_minus_beta1, out=denom)
            m += denom

            np.square(grad, out=scratch)
            scratch *= one_minus_beta2
            v *= beta2
            v += scratch

            np.divide(v, beta2_correction, out=denom)
            np.sqrt(denom, out=denom)
            denom += eps
            np.divide(m, beta1_correction, out=scratch)
            scratch /= denom
            param -= learning_rate * scratch
            if not np.all(np.isfinite(param)):
                raise FloatingPointError(f"Parameter {name} became non-finite during training.")

        return total_norm

    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        grad_clip: float = 1.0,
    ) -> tuple[float, float]:
        loss, grads = self.loss_and_grads(x, y)
        grad_norm = self.apply_grads(grads, learning_rate=learning_rate, grad_clip=grad_clip)
        return loss, grad_norm

    def generate(
        self,
        corpus: Corpus,
        prompt: str = "",
        length: int = 400,
        temperature: float = 0.9,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> str:
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        sample_rng = np.random.default_rng(seed)
        context = [corpus.pad_id] * self.context_size
        output = list(prompt)

        for token_id in corpus.encode_prompt(prompt):
            context = context[1:] + [token_id]

        for _ in range(length):
            x = np.array([context], dtype=np.int32)
            logits, _ = self.forward(x)
            scores = logits[0] / temperature

            if top_k is not None and 0 < top_k < len(scores):
                cutoff = np.partition(scores, -top_k)[-top_k]
                scores = np.where(scores < cutoff, -1e9, scores)

            shifted = scores - scores.max()
            probs = np.exp(shifted)
            probs /= probs.sum()

            next_id = int(sample_rng.choice(len(probs), p=probs))
            context = context[1:] + [next_id]
            if next_id != corpus.pad_id:
                output.append(corpus.itos[next_id])

        return "".join(output)

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "params": {name: value.copy() for name, value in self.params.items()},
            "opt_state": {
                "step": self.opt_state["step"],
                "m": {name: value.copy() for name, value in self.opt_state["m"].items()},
                "v": {name: value.copy() for name, value in self.opt_state["v"].items()},
            },
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> "CharMLP":
        model = cls(
            vocab_size=payload["vocab_size"],
            context_size=payload["context_size"],
            embed_dim=payload["embed_dim"],
            hidden_dim=payload["hidden_dim"],
        )
        model.params = {name: value.copy() for name, value in payload["params"].items()}
        model.opt_state = {
            "step": payload["opt_state"]["step"],
            "m": {name: value.copy() for name, value in payload["opt_state"]["m"].items()},
            "v": {name: value.copy() for name, value in payload["opt_state"]["v"].items()},
        }
        model._scratch = {name: np.empty_like(value) for name, value in model.params.items()}
        model._denom_scratch = {name: np.empty_like(value) for name, value in model.params.items()}
        return model


class TextGeneratorTrainer:
    def __init__(
        self,
        corpus: Corpus,
        model: CharMLP,
        batch_size: int = 64,
        learning_rate: float = 3e-3,
        grad_clip: float = 1.0,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if grad_clip <= 0:
            raise ValueError("grad_clip must be positive.")

        self.corpus = corpus
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.seed = seed
        self.train_steps = 0
        self.rng = np.random.default_rng(seed)
        self._batch_offsets = np.arange(self.corpus.context_size, dtype=np.int32)

    def batch(self, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
        if split == "train":
            padded = self.corpus.train_padded
            data_length = len(self.corpus.train_encoded)
        elif split == "val":
            padded = self.corpus.val_padded
            data_length = len(self.corpus.val_encoded)
        else:
            raise ValueError(f"Unknown split: {split}")

        starts = self.rng.integers(0, data_length, size=self.batch_size, endpoint=False)
        x = padded[starts[:, None] + self._batch_offsets]
        y = padded[starts + self.corpus.context_size]
        return x.astype(np.int32, copy=False), y.astype(np.int32, copy=False)

    def train(
        self,
        steps: int,
        log_every: int = 100,
    ) -> None:
        if log_every <= 0:
            raise ValueError("log_every must be a positive integer.")

        recent_losses: deque[float] = deque(maxlen=log_every)
        recent_loss_sum = 0.0

        try:
            for _ in range(steps):
                x, y = self.batch("train")
                loss, grad_norm = self.model.train_step(
                    x,
                    y,
                    learning_rate=self.learning_rate,
                    grad_clip=self.grad_clip,
                )
                self.train_steps += 1
                if len(recent_losses) == log_every:
                    recent_loss_sum -= recent_losses[0]
                recent_losses.append(loss)
                recent_loss_sum += loss

                if self.train_steps % log_every == 0:
                    avg_loss = recent_loss_sum / len(recent_losses)
                    val_loss = self.estimate_loss("val", num_batches=4)
                    print(
                        f"step {self.train_steps:>7} | train_loss={avg_loss:.4f} "
                        f"| val_loss={val_loss:.4f} | grad_norm={grad_norm:.4f}"
                    )
        except KeyboardInterrupt:
            print("\nTraining interrupted. Current model state is still available in memory.")

    def estimate_loss(self, split: str = "val", num_batches: int = 8) -> float:
        losses = []
        for _ in range(num_batches):
            x, y = self.batch(split)
            losses.append(self.model.loss(x, y))
        return float(sum(losses) / len(losses))

    def status(self) -> str:
        train_loss = self.estimate_loss("train", num_batches=4)
        val_loss = self.estimate_loss("val", num_batches=4)
        return (
            f"steps={self.train_steps}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"batch_size={self.batch_size}, lr={self.learning_rate}, context={self.corpus.context_size}, "
            f"hidden_dim={self.model.hidden_dim}, embed_dim={self.model.embed_dim}, "
            f"vocab={self.corpus.vocab_size}"
        )

    def config(self) -> str:
        return (
            f"steps={self.train_steps}, batch_size={self.batch_size}, lr={self.learning_rate}, "
            f"grad_clip={self.grad_clip}, context={self.corpus.context_size}, "
            f"hidden_dim={self.model.hidden_dim}, embed_dim={self.model.embed_dim}, "
            f"vocab={self.corpus.vocab_size}, seed={self.seed}"
        )

    def save(self, path: Path, data_path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_path": str(data_path),
            "corpus_sha256": corpus_fingerprint(self.corpus.text),
            "chars": self.corpus.chars,
            "trainer": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "grad_clip": self.grad_clip,
                "seed": self.seed,
                "train_steps": self.train_steps,
                "rng_state": self.rng.bit_generator.state,
            },
            "model": self.model.to_checkpoint(),
        }
        param_names = sorted(payload["model"]["params"])
        metadata = {
            "format": "text_generator_checkpoint_v2",
            "data_path": payload["data_path"],
            "corpus_sha256": payload["corpus_sha256"],
            "chars": payload["chars"],
            "trainer": payload["trainer"],
            "model": {
                "vocab_size": payload["model"]["vocab_size"],
                "context_size": payload["model"]["context_size"],
                "embed_dim": payload["model"]["embed_dim"],
                "hidden_dim": payload["model"]["hidden_dim"],
                "opt_step": payload["model"]["opt_state"]["step"],
                "param_names": param_names,
            },
        }
        archive_items: dict[str, Any] = {"metadata_json": np.array(json.dumps(metadata))}
        for name in param_names:
            archive_items[f"param__{name}"] = payload["model"]["params"][name]
            archive_items[f"opt_m__{name}"] = payload["model"]["opt_state"]["m"][name]
            archive_items[f"opt_v__{name}"] = payload["model"]["opt_state"]["v"][name]

        with path.open("wb") as handle:
            np.savez(handle, **archive_items)
        print(f"Saved checkpoint to {path}")

    @staticmethod
    def load_safe_checkpoint(path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive:
                raise ValueError("Checkpoint is missing metadata and cannot be loaded.")
            try:
                metadata = json.loads(str(archive["metadata_json"].tolist()))
            except json.JSONDecodeError as exc:
                raise ValueError("Checkpoint metadata is corrupted and cannot be loaded.") from exc

            if metadata.get("format") != "text_generator_checkpoint_v2":
                raise ValueError("Checkpoint format is not recognized.")

            model_meta = metadata["model"]
            param_names = model_meta["param_names"]
            params = {name: archive[f"param__{name}"].copy() for name in param_names}
            opt_state_m = {name: archive[f"opt_m__{name}"].copy() for name in param_names}
            opt_state_v = {name: archive[f"opt_v__{name}"].copy() for name in param_names}

        return {
            "data_path": metadata["data_path"],
            "corpus_sha256": metadata["corpus_sha256"],
            "chars": metadata["chars"],
            "trainer": metadata["trainer"],
            "model": {
                "vocab_size": model_meta["vocab_size"],
                "context_size": model_meta["context_size"],
                "embed_dim": model_meta["embed_dim"],
                "hidden_dim": model_meta["hidden_dim"],
                "params": params,
                "opt_state": {
                    "step": model_meta["opt_step"],
                    "m": opt_state_m,
                    "v": opt_state_v,
                },
            },
        }

    @classmethod
    def load(
        cls,
        path: Path,
        corpus: Corpus,
        allow_unsafe_checkpoint: bool = False,
    ) -> "TextGeneratorTrainer":
        path = path.expanduser().resolve()

        if zipfile.is_zipfile(path):
            payload = cls.load_safe_checkpoint(path)
        else:
            if not allow_unsafe_checkpoint:
                raise ValueError(
                    "Refusing to load a legacy pickle checkpoint without --allow-unsafe-checkpoint. "
                    "Legacy pickle checkpoints can execute arbitrary code."
                )
            with path.open("rb") as handle:
                payload = pickle.load(handle)

        if payload["chars"] != corpus.chars:
            raise ValueError(
                "Checkpoint vocabulary does not match the current training text. "
                "Use the same shakespeare.txt that created the checkpoint."
            )
        if payload["model"]["context_size"] != corpus.context_size:
            raise ValueError(
                "Checkpoint context size does not match the current configuration. "
                "Use the same --context-size value, or start with --fresh."
            )
        checkpoint_corpus_hash = payload.get("corpus_sha256")
        if checkpoint_corpus_hash is None:
            raise ValueError(
                "Legacy checkpoint is missing a corpus fingerprint and can no longer be loaded safely."
            )
        if checkpoint_corpus_hash != corpus_fingerprint(corpus.text):
            raise ValueError(
                "Checkpoint training text does not match the current corpus contents. "
                "Use the same training text that created the checkpoint, or start with --fresh."
            )

        model = CharMLP.from_checkpoint(payload["model"])
        trainer = cls(
            corpus=corpus,
            model=model,
            batch_size=payload["trainer"]["batch_size"],
            learning_rate=payload["trainer"]["learning_rate"],
            grad_clip=payload["trainer"]["grad_clip"],
            seed=payload["trainer"].get("seed", 42),
        )
        trainer.train_steps = payload["trainer"]["train_steps"]
        trainer.rng.bit_generator.state = payload["trainer"]["rng_state"]
        return trainer


def build_new_trainer(
    data_path: Path,
    *,
    context_size: int,
    embed_dim: int,
    hidden_dim: int,
    batch_size: int,
    learning_rate: float,
    grad_clip: float,
    seed: int,
) -> TextGeneratorTrainer:
    text = read_text(data_path)
    corpus = Corpus(text=text, context_size=context_size)
    model = CharMLP(
        vocab_size=corpus.vocab_size,
        context_size=corpus.context_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    return TextGeneratorTrainer(
        corpus=corpus,
        model=model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        grad_clip=grad_clip,
        seed=seed,
    )


def build_new_trainer_from_args(args: argparse.Namespace, data_path: Path) -> TextGeneratorTrainer:
    return build_new_trainer(
        data_path,
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )


def load_trainer_from_checkpoint(
    checkpoint_path: Path,
    *,
    data_path: Path,
    context_size: int,
    allow_unsafe_checkpoint: bool,
) -> TextGeneratorTrainer:
    text = read_text(data_path)
    corpus = Corpus(text=text, context_size=context_size)
    return TextGeneratorTrainer.load(
        checkpoint_path,
        corpus,
        allow_unsafe_checkpoint=allow_unsafe_checkpoint,
    )


def build_trainer_for_target(
    args: argparse.Namespace,
    *,
    data_path: Path,
    target: RunTarget,
    require_existing_checkpoint: bool = False,
) -> tuple[TextGeneratorTrainer, Path | None]:
    if args.fresh:
        return build_new_trainer_from_args(args, data_path), None

    if target.checkpoint_path.exists():
        return (
            load_trainer_from_checkpoint(
                target.checkpoint_path,
                data_path=data_path,
                context_size=args.context_size,
                allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
            ),
            target.checkpoint_path,
        )

    if require_existing_checkpoint:
        if target.managed:
            raise FileNotFoundError(
                f"No saved progress found for session '{target.session_name}'. "
                "Train that session first, or pass --fresh."
            )
        raise FileNotFoundError(
            f"No checkpoint found at {target.checkpoint_path}. "
            "Train a model first, or pass --fresh."
        )

    return build_new_trainer_from_args(args, data_path), None


def doctor_report(
    data_path: Path,
    target: RunTarget | None,
    starter_checkpoint_path: Path | None,
    session_manager: SessionManager,
) -> list[str]:
    session_names = session_manager.list_sessions()
    report = [
        f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"NumPy: {np.__version__}",
        f"Training text: {'OK' if data_path.exists() else 'MISSING'} ({data_path})",
        f"Sessions directory: {session_manager.sessions_root}",
        f"Managed sessions: {', '.join(session_names) if session_names else 'none'}",
    ]
    if target is None:
        report.append("Active target: none selected")
    elif target.managed:
        checkpoint_status = "FOUND" if target.checkpoint_path.exists() else "NOT FOUND"
        report.append(
            f"Active session: {target.session_name} ({checkpoint_status}) "
            f"({target.checkpoint_path})"
        )
    else:
        checkpoint_status = "FOUND" if target.checkpoint_path.exists() else "NOT FOUND"
        report.append(f"Explicit checkpoint: {checkpoint_status} ({target.checkpoint_path})")
    if starter_checkpoint_path is None:
        report.append("Starter checkpoint: NOT FOUND")
    else:
        report.append(f"Starter checkpoint: FOUND ({starter_checkpoint_path})")
    return report


def show_doctor_report(
    data_path: Path,
    target: RunTarget | None,
    starter_checkpoint_path: Path | None,
    session_manager: SessionManager,
) -> None:
    for line in doctor_report(
        data_path,
        target,
        starter_checkpoint_path,
        session_manager,
    ):
        print(line)


def parse_shell_sample_args(args: list[str], *, default_length: int) -> tuple[int, str]:
    length = default_length
    prompt_args = args
    if args:
        try:
            length = int(args[0])
            prompt_args = args[1:]
        except ValueError:
            prompt_args = args
    return length, " ".join(prompt_args)


def run_sample_command(
    trainer: TextGeneratorTrainer,
    *,
    prompt: str,
    length: int,
    temperature: float = 0.9,
    top_k: int | None = None,
) -> None:
    clean_prompt, notes = sanitize_prompt(prompt, trainer.corpus)
    for note in notes:
        print(f"Note: {note}")
    print(
        trainer.model.generate(
            trainer.corpus,
            prompt=clean_prompt,
            length=length,
            temperature=temperature,
            top_k=top_k,
        )
    )


def train_and_save(
    trainer: TextGeneratorTrainer,
    *,
    steps: int,
    data_path: Path,
    checkpoint_path: Path,
    log_every: int = 100,
) -> None:
    trainer.train(steps=steps, log_every=log_every)
    trainer.save(checkpoint_path, data_path)


def describe_start_source(target: RunTarget, source_checkpoint_path: Path | None) -> str:
    if source_checkpoint_path is None:
        return "a fresh random model"
    starter_checkpoint_path = resolve_starter_checkpoint_path()
    if starter_checkpoint_path is not None and source_checkpoint_path == starter_checkpoint_path:
        return "the bundled starter checkpoint"
    if target.managed and source_checkpoint_path == target.checkpoint_path:
        return f"session '{target.session_name}'"
    return f"the checkpoint at {source_checkpoint_path}"


def configure_shell_readline() -> bool:
    try:
        import readline
    except ImportError:
        return False

    backend = getattr(readline, "backend", "")
    if not backend:
        doc = (getattr(readline, "__doc__", "") or "").lower()
        backend = "editline" if "libedit" in doc else "readline"
    binding = "bind ^L ed-clear-screen" if backend == "editline" else '"\\C-l": clear-screen'
    try:
        readline.parse_and_bind(binding)
    except Exception:  # noqa: BLE001
        return False
    return True


def is_clear_screen_request(raw_line: str) -> bool:
    return "\x0c" in raw_line and raw_line.replace("\x0c", "").strip() == ""


def clear_terminal_screen() -> None:
    if not sys.stdout.isatty():
        return
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


class InteractiveShell:
    prompt = "textgen> "

    def __init__(
        self,
        trainer: TextGeneratorTrainer,
        data_path: Path,
        run_target: RunTarget,
        session_manager: SessionManager,
        *,
        allow_unsafe_checkpoint: bool = False,
        source_checkpoint_path: Path | None = None,
    ) -> None:
        self.trainer = trainer
        self.data_path = data_path
        self.run_target = run_target
        self.session_manager = session_manager
        self.allow_unsafe_checkpoint = allow_unsafe_checkpoint
        self.source_checkpoint_path = source_checkpoint_path

    def intro_lines(self) -> list[str]:
        if self.run_target.managed:
            target_line = f"Current session: {self.run_target.session_name}"
        else:
            target_line = f"Explicit checkpoint target: {self.run_target.checkpoint_path}"
        return [
            "Shakespeare text lab",
            target_line,
            f"Starting from {describe_start_source(self.run_target, self.source_checkpoint_path)}.",
            "Type 'help' for commands.",
        ]

    def command_handlers(self) -> dict[str, Callable[[list[str]], bool | None]]:
        return {
            "help": self.handle_help,
            "?": self.handle_help,
            "status": self.handle_status,
            "config": self.handle_config,
            "doctor": self.handle_doctor,
            "session": self.handle_session,
            "sessions": self.handle_sessions,
            "train": self.handle_train,
            "sample": self.handle_sample,
            "save": self.handle_save,
            "load": self.handle_load,
            "rebuild": self.handle_rebuild,
            "rebuild-model": self.handle_rebuild,
            "rebuild_model": self.handle_rebuild,
            "reset": self.handle_reset,
            "delete-session": self.handle_delete_session,
            "delete_session": self.handle_delete_session,
            "quit": self.handle_quit,
            "exit": self.handle_quit,
        }

    def cmdloop(self) -> None:
        configure_shell_readline()
        for line in self.intro_lines():
            print(line)

        while True:
            try:
                raw_line = input(self.prompt)
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue

            if is_clear_screen_request(raw_line):
                clear_terminal_screen()
                continue

            line = raw_line.strip()
            if not line:
                continue

            if not self.execute(line):
                break

    def execute(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            print(f"Could not parse command: {exc}")
            return True

        command = parts[0].lower()
        args = parts[1:]
        handler = self.command_handlers().get(command)
        if handler is None:
            print(f"Unknown command: {command}")
            return True

        try:
            result = handler(args)
        except Exception as exc:  # noqa: BLE001
            print(f"Command failed: {exc}")
            return True

        return result is not False

    def handle_help(self, args: list[str]) -> None:
        self.print_help()

    def handle_status(self, args: list[str]) -> None:
        print(self.trainer.status())

    def handle_config(self, args: list[str]) -> None:
        print(self.trainer.config())

    def handle_doctor(self, args: list[str]) -> None:
        show_doctor_report(
            self.data_path,
            self.run_target,
            resolve_starter_checkpoint_path(),
            self.session_manager,
        )

    def handle_session(self, args: list[str]) -> None:
        if self.run_target.managed:
            print(f"Current session: {self.run_target.session_name}")
            print(f"Checkpoint: {self.run_target.checkpoint_path}")
        else:
            print("Current target: explicit checkpoint")
            print(f"Checkpoint: {self.run_target.checkpoint_path}")

    def handle_sessions(self, args: list[str]) -> None:
        session_names = self.session_manager.list_sessions()
        if not session_names:
            print("No managed sessions found.")
            return

        for name in session_names:
            tags: list[str] = []
            if self.run_target.managed and self.run_target.session_name == name:
                tags.extend(["current", "locked"])
            elif self.session_manager.is_locked(name):
                tags.append("locked")
            if tags:
                print(f"{name} ({', '.join(tags)})")
            else:
                print(name)

    def handle_train(self, args: list[str]) -> None:
        steps = int(args[0]) if args else DEFAULT_SHELL_TRAIN_STEPS
        if steps <= 0:
            raise ValueError("steps must be a positive integer")
        train_and_save(
            self.trainer,
            steps=steps,
            data_path=self.data_path,
            checkpoint_path=self.run_target.checkpoint_path,
        )
        self.source_checkpoint_path = self.run_target.checkpoint_path

    def handle_sample(self, args: list[str]) -> None:
        length, prompt = parse_shell_sample_args(args, default_length=DEFAULT_SHELL_SAMPLE_LENGTH)
        if length <= 0:
            raise ValueError("sample length must be a positive integer")
        run_sample_command(
            self.trainer,
            prompt=prompt,
            length=length,
        )

    def handle_save(self, args: list[str]) -> None:
        if len(args) > 1:
            raise ValueError("Usage: save [path]")
        path = resolve_checkpoint_argument(args[0]) if args else self.run_target.checkpoint_path
        if path == self.run_target.checkpoint_path:
            self.trainer.save(path, self.data_path)
            return
        with acquire_exclusive_lock(explicit_checkpoint_lock_path(path), f"Checkpoint {path}"):
            self.trainer.save(path, self.data_path)

    def handle_load(self, args: list[str]) -> None:
        allow_unsafe_checkpoint = self.allow_unsafe_checkpoint
        path_args: list[str] = []
        for arg in args:
            if arg == "--unsafe":
                allow_unsafe_checkpoint = True
            else:
                path_args.append(arg)
        if len(path_args) > 1:
            raise ValueError("Usage: load [path] [--unsafe]")
        path = resolve_checkpoint_argument(path_args[0]) if path_args else self.run_target.checkpoint_path
        if path == self.run_target.checkpoint_path:
            self.trainer = TextGeneratorTrainer.load(
                path,
                self.trainer.corpus,
                allow_unsafe_checkpoint=allow_unsafe_checkpoint,
            )
        else:
            with acquire_exclusive_lock(explicit_checkpoint_lock_path(path), f"Checkpoint {path}"):
                self.trainer = TextGeneratorTrainer.load(
                    path,
                    self.trainer.corpus,
                    allow_unsafe_checkpoint=allow_unsafe_checkpoint,
                )
        self.source_checkpoint_path = path
        print(f"Loaded checkpoint from {path}")

    def handle_rebuild(self, args: list[str]) -> None:
        if len(args) not in {3, 5}:
            print(
                "Usage: rebuild <context_size> <hidden_dim> <learning_rate> "
                "[batch_size] [embed_dim]"
            )
            return

        self.rebuild_model(
            context_size=int(args[0]),
            hidden_dim=int(args[1]),
            learning_rate=float(args[2]),
            batch_size=int(args[3]) if len(args) == 5 else None,
            embed_dim=int(args[4]) if len(args) == 5 else None,
        )

    def handle_reset(self, args: list[str]) -> None:
        if not self.run_target.managed or self.run_target.session_name is None:
            raise ValueError("reset is only available for managed sessions.")

        checkpoint_path = self.run_target.checkpoint_path
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        self.trainer = build_new_trainer(
            self.data_path,
            context_size=self.trainer.corpus.context_size,
            embed_dim=self.trainer.model.embed_dim,
            hidden_dim=self.trainer.model.hidden_dim,
            batch_size=self.trainer.batch_size,
            learning_rate=self.trainer.learning_rate,
            grad_clip=self.trainer.grad_clip,
            seed=self.trainer.seed,
        )
        self.source_checkpoint_path = None
        print(
            f"Reset complete. Session '{self.run_target.session_name}' is back to a fresh random model."
        )

    def handle_delete_session(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("Usage: delete-session <name>")
        current_session = self.run_target.session_name if self.run_target.managed else None
        print(self.session_manager.delete_session(args[0], current_session=current_session))

    def handle_quit(self, args: list[str]) -> bool:
        return False

    def rebuild_model(
        self,
        context_size: int,
        hidden_dim: int,
        learning_rate: float,
        batch_size: int | None = None,
        embed_dim: int | None = None,
    ) -> None:
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if embed_dim is not None and embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        text = read_text(self.data_path)
        corpus = Corpus(text=text, context_size=context_size)
        model = CharMLP(
            vocab_size=corpus.vocab_size,
            context_size=corpus.context_size,
            embed_dim=embed_dim if embed_dim is not None else self.trainer.model.embed_dim,
            hidden_dim=hidden_dim,
            seed=self.trainer.seed,
        )
        self.trainer = TextGeneratorTrainer(
            corpus=corpus,
            model=model,
            batch_size=batch_size if batch_size is not None else self.trainer.batch_size,
            learning_rate=learning_rate,
            grad_clip=self.trainer.grad_clip,
            seed=self.trainer.seed,
        )
        batch_summary = self.trainer.batch_size
        embed_summary = self.trainer.model.embed_dim
        print(
            "Rebuilt in-memory model with "
            f"context={context_size}, hidden_dim={hidden_dim}, lr={learning_rate}, "
            f"batch_size={batch_summary}, embed_dim={embed_summary}. "
            "Training steps reset to 0. Save or train to write a new checkpoint."
        )

    @staticmethod
    def print_help() -> None:
        print("Commands:")
        print("  help                     Show this message")
        print("  config                   Show current model/training settings")
        print("  status                   Estimate current train/validation loss")
        print("  doctor                   Check package and session setup")
        print("  session                  Show the current active session or checkpoint")
        print("  sessions                 List managed sessions")
        print("  train [steps]            Train for more steps, then autosave")
        print("  sample [len] [prompt]    Generate text, optionally seeded by a prompt")
        print("  save [path]              Save a checkpoint")
        print("  load [path] [--unsafe]   Load a checkpoint")
        print("                           --unsafe is only required for legacy pickle checkpoints")
        print("  rebuild c h lr [b] [e]   Rebuild model with new context, hidden dim, lr,")
        print("                           and optionally batch size and embed dim")
        print(
            "                           Limits: context_size >= 1, hidden_dim >= 1, "
            "batch_size >= 1, embed_dim >= 1, lr > 0"
        )
        print("  reset                    Clear the current managed session and start fresh")
        print("  delete-session <name>    Delete a managed session that is not current or locked")
        print("  Ctrl+L                   Clear the screen at the prompt")
        print("  quit                     Exit the shell")


def resolve_target_from_args(args: argparse.Namespace, session_manager: SessionManager) -> RunTarget | None:
    if args.checkpoint is not None:
        return session_manager.target_for_checkpoint(resolve_checkpoint_argument(args.checkpoint))
    if args.session is not None:
        return session_manager.target_for_session(args.session)
    return None


def resolve_shell_target(args: argparse.Namespace, session_manager: SessionManager) -> RunTarget:
    target = resolve_target_from_args(args, session_manager)
    if target is not None:
        return target
    session_name = session_manager.choose_session_interactively()
    return session_manager.target_for_session(session_name)


def require_noninteractive_target(
    args: argparse.Namespace,
    *,
    session_manager: SessionManager,
    command_name: str,
) -> RunTarget:
    target = resolve_target_from_args(args, session_manager)
    if target is not None:
        return target

    if command_name == "train":
        raise ValueError(
            "train requires --session or --checkpoint so it knows where to save progress."
        )
    raise ValueError(
        "sample requires --session, --checkpoint, or --fresh."
    )


def normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["shell"]
    if argv[0].startswith("-"):
        return ["shell", *argv]
    return argv


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple character-level text generator trained on shakespeare.txt"
    )
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    common.add_argument("--checkpoint", default=None)
    common.add_argument("--session", default=None)
    common.add_argument("--context-size", type=parse_positive_int, default=48)
    common.add_argument("--embed-dim", type=parse_positive_int, default=24)
    common.add_argument("--hidden-dim", type=parse_positive_int, default=192)
    common.add_argument("--batch-size", type=parse_positive_int, default=64)
    common.add_argument("--learning-rate", type=parse_positive_float, default=3e-3)
    common.add_argument("--grad-clip", type=parse_positive_float, default=1.0)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="Allow loading a legacy pickle checkpoint that may execute arbitrary code",
    )
    common.add_argument(
        "--fresh",
        action="store_true",
        help="Start from a new model even if a checkpoint already exists",
    )

    shell_parser = subparsers.add_parser("shell", parents=[common], help="Interactive shell")
    shell_parser.set_defaults(command="shell")

    train_parser = subparsers.add_parser("train", parents=[common], help="Train for a fixed number of steps")
    train_parser.add_argument("--steps", type=parse_positive_int, default=1000)
    train_parser.add_argument("--log-every", type=parse_positive_int, default=100)

    sample_parser = subparsers.add_parser("sample", parents=[common], help="Generate text from a checkpoint")
    sample_parser.add_argument("--length", type=parse_positive_int, default=300)
    sample_parser.add_argument("--prompt", default="")
    sample_parser.add_argument("--temperature", type=parse_positive_float, default=0.9)
    sample_parser.add_argument("--top-k", type=parse_positive_int, default=12)

    doctor_parser = subparsers.add_parser("doctor", parents=[common], help="Check the local package setup")
    doctor_parser.set_defaults(command="doctor")

    return parser


def main() -> int:
    parser = make_parser()
    argv = normalize_argv(sys.argv[1:])
    args = parser.parse_args(argv)

    session_manager = SessionManager()
    migration_messages = session_manager.migrate_legacy_checkpoint()
    for message in migration_messages:
        print(message)

    data_path = Path(args.data).expanduser().resolve()

    if args.command == "doctor":
        target = resolve_target_from_args(args, session_manager)
        show_doctor_report(
            data_path,
            target,
            resolve_starter_checkpoint_path(),
            session_manager,
        )
        return 0

    if args.command == "shell":
        try:
            target = resolve_shell_target(args, session_manager)
            with acquire_target_lock(target):
                trainer, source_checkpoint_path = build_trainer_for_target(
                    args,
                    data_path=data_path,
                    target=target,
                )
                shell = InteractiveShell(
                    trainer=trainer,
                    data_path=data_path,
                    run_target=target,
                    session_manager=session_manager,
                    allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
                    source_checkpoint_path=source_checkpoint_path,
                )
                shell.cmdloop()
        except (EOFError, FileNotFoundError, SessionError, SessionLockError, ValueError) as exc:
            parser.exit(2, f"{parser.prog}: error: {exc}\n")
        return 0

    if args.command == "train":
        try:
            target = require_noninteractive_target(
                args,
                session_manager=session_manager,
                command_name="train",
            )
            with acquire_target_lock(target):
                trainer, _ = build_trainer_for_target(
                    args,
                    data_path=data_path,
                    target=target,
                )
                train_and_save(
                    trainer,
                    steps=args.steps,
                    data_path=data_path,
                    checkpoint_path=target.checkpoint_path,
                    log_every=args.log_every,
                )
                print(trainer.status())
        except (FileNotFoundError, SessionError, SessionLockError, ValueError) as exc:
            parser.exit(2, f"{parser.prog}: error: {exc}\n")
        return 0

    if args.command == "sample":
        try:
            if args.fresh:
                trainer = build_new_trainer_from_args(args, data_path)
            else:
                target = require_noninteractive_target(
                    args,
                    session_manager=session_manager,
                    command_name="sample",
                )
                with acquire_target_lock(target):
                    trainer, _ = build_trainer_for_target(
                        args,
                        data_path=data_path,
                        target=target,
                        require_existing_checkpoint=True,
                    )
                    run_sample_command(
                        trainer,
                        prompt=args.prompt,
                        length=args.length,
                        temperature=args.temperature,
                        top_k=args.top_k,
                    )
                    return 0

            run_sample_command(
                trainer,
                prompt=args.prompt,
                length=args.length,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        except (FileNotFoundError, SessionError, SessionLockError, ValueError) as exc:
            parser.exit(2, f"{parser.prog}: error: {exc}\n")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

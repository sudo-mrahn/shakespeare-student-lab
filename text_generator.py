#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shlex
import sys
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINTS_DIR = Path(__file__).with_name("checkpoints")
DEFAULT_DATA_PATH = Path(__file__).with_name("shakespeare.txt")
DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "latest_model.npz"
DEFAULT_STARTER_CHECKPOINT_PATH = CHECKPOINTS_DIR / "starter_model.npz"
CHECKPOINT_SUFFIXES = {".npz", ".pkl", ".pickle"}
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
STUDENT_DEFAULT_SAMPLE_LENGTH = 250
STUDENT_DEFAULT_TRAIN_STEPS = 200
STUDENT_MAX_SAMPLE_LENGTH = 1000
STUDENT_MAX_TRAIN_STEPS = 5000


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


def is_checkpoint_file(path: Path) -> bool:
    return path.suffix.lower() in CHECKPOINT_SUFFIXES


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


def resolve_checkpoint_argument(checkpoint: str | None, default_path: Path) -> Path:
    if checkpoint is None:
        return default_path.expanduser().resolve()
    return Path(checkpoint).expanduser().resolve()


def choose_sample_checkpoint_path(
    checkpoint: str | None,
    *,
    allow_unsafe_checkpoint: bool = False,
    work_checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> Path:
    if checkpoint is not None:
        return Path(checkpoint).expanduser().resolve()
    return work_checkpoint_path.expanduser().resolve()


def student_checkpoint_path() -> Path:
    return DEFAULT_CHECKPOINT_PATH.expanduser().resolve()


def starter_checkpoint_candidates() -> list[Path]:
    return [DEFAULT_STARTER_CHECKPOINT_PATH.expanduser().resolve()]


def resolve_starter_checkpoint_path() -> Path | None:
    safe_path = DEFAULT_STARTER_CHECKPOINT_PATH.expanduser().resolve()
    if safe_path.exists():
        return safe_path
    return None


def describe_checkpoint_source(path: Path | None) -> str:
    if path is None:
        return "a fresh random model"
    resolved = path.expanduser().resolve()
    if resolved == DEFAULT_CHECKPOINT_PATH.expanduser().resolve():
        return "your saved progress"
    if resolved == DEFAULT_STARTER_CHECKPOINT_PATH.expanduser().resolve():
        return "the bundled starter model"
    return f"the checkpoint at {resolved}"


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
        grads["E"] = np.zeros_like(self.params["E"])
        np.add.at(grads["E"], cache["x"], dembedding)

        if not np.isfinite(loss):
            raise FloatingPointError("Loss became non-finite during training.")
        for name, grad in grads.items():
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError(f"Gradient {name} became non-finite during training.")

        return loss, grads

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

        for name, param in self.params.items():
            grad = grads[name] * clip_scale
            self.opt_state["m"][name] = beta1 * self.opt_state["m"][name] + (1.0 - beta1) * grad
            self.opt_state["v"][name] = beta2 * self.opt_state["v"][name] + (1.0 - beta2) * (
                grad * grad
            )

            m_hat = self.opt_state["m"][name] / (1.0 - beta1**step)
            v_hat = self.opt_state["v"][name] / (1.0 - beta2**step)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
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
        offsets = np.arange(self.corpus.context_size, dtype=np.int32)
        x = padded[starts[:, None] + offsets]
        y = padded[starts + self.corpus.context_size]
        return x.astype(np.int32, copy=False), y.astype(np.int32, copy=False)

    def train(
        self,
        steps: int,
        log_every: int = 100,
    ) -> None:
        if log_every <= 0:
            raise ValueError("log_every must be a positive integer.")

        recent_losses: list[float] = []

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
                recent_losses.append(loss)

                if self.train_steps % log_every == 0:
                    avg_loss = sum(recent_losses[-log_every:]) / min(len(recent_losses), log_every)
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
            np.savez_compressed(handle, **archive_items)
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
        data_path: Path | None = None,
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
        data_path,
        allow_unsafe_checkpoint=allow_unsafe_checkpoint,
    )


def build_student_trainer(args: argparse.Namespace) -> tuple[TextGeneratorTrainer, Path | None]:
    data_path = Path(args.data).expanduser().resolve()
    work_checkpoint_path = resolve_checkpoint_argument(args.checkpoint, DEFAULT_CHECKPOINT_PATH)

    if args.fresh:
        return (
            build_new_trainer(
                data_path,
                context_size=args.context_size,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                grad_clip=args.grad_clip,
                seed=args.seed,
            ),
            None,
        )

    if work_checkpoint_path.exists():
        return (
            load_trainer_from_checkpoint(
                work_checkpoint_path,
                data_path=data_path,
                context_size=args.context_size,
                allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
            ),
            work_checkpoint_path,
        )

    return (
        build_new_trainer(
            data_path,
            context_size=args.context_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            seed=args.seed,
        ),
        None,
    )


def doctor_report(
    data_path: Path,
    work_checkpoint_path: Path,
    starter_checkpoint_path: Path | None,
) -> list[str]:
    report = [
        f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"NumPy: {np.__version__}",
        f"Training text: {'OK' if data_path.exists() else 'MISSING'} ({data_path})",
        (
            f"Saved progress checkpoint: {'FOUND' if work_checkpoint_path.exists() else 'NOT FOUND'} "
            f"({work_checkpoint_path})"
        ),
    ]
    if starter_checkpoint_path is None:
        report.append("Starter checkpoint: NOT FOUND")
    else:
        report.append(f"Starter checkpoint: FOUND ({starter_checkpoint_path})")
    return report


class StudentShell:
    def __init__(
        self,
        trainer: TextGeneratorTrainer,
        data_path: Path,
        checkpoint_path: Path,
        starter_checkpoint_path: Path | None,
        source_checkpoint_path: Path | None,
        allow_unsafe_checkpoint: bool = False,
    ) -> None:
        self.trainer = trainer
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.starter_checkpoint_path = starter_checkpoint_path
        self.source_checkpoint_path = source_checkpoint_path
        self.allow_unsafe_checkpoint = allow_unsafe_checkpoint

    def cmdloop(self) -> None:
        print("Student-friendly Shakespeare text lab")
        print(f"Starting from {describe_checkpoint_source(self.source_checkpoint_path)}.")
        print("Try: sample 500")
        print("Then: train 100")
        print("Type 'help' for the small command list.")

        while True:
            try:
                line = input("student> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue

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

        try:
            if command in {"help", "?"}:
                self.print_help()
            elif command == "status":
                print(self.trainer.status())
            elif command == "config":
                print(self.trainer.config())
            elif command == "sample":
                self.sample(args)
            elif command == "train":
                self.train(args)
            elif command in {"reset", "starter"}:
                self.reset_to_fresh()
            elif command == "doctor":
                self.print_doctor_report()
            elif command in {"quit", "exit"}:
                return False
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' to see the student command list.")
        except Exception as exc:  # noqa: BLE001
            print(f"Command failed: {exc}")

        return True

    def train(self, args: list[str]) -> None:
        steps = STUDENT_DEFAULT_TRAIN_STEPS
        if args:
            steps = int(args[0])
        if steps <= 0:
            raise ValueError("steps must be a positive integer")
        if steps > STUDENT_MAX_TRAIN_STEPS:
            raise ValueError(
                f"steps is capped at {STUDENT_MAX_TRAIN_STEPS} in student mode to keep runs manageable"
            )

        self.trainer.train(steps=steps)
        self.trainer.save(self.checkpoint_path, self.data_path)
        self.source_checkpoint_path = self.checkpoint_path
        print(f"Saved your progress to {self.checkpoint_path}")

    def sample(self, args: list[str]) -> None:
        length = STUDENT_DEFAULT_SAMPLE_LENGTH
        prompt_args = args
        if args:
            try:
                length = int(args[0])
                prompt_args = args[1:]
            except ValueError:
                prompt_args = args

        if length <= 0:
            raise ValueError("sample length must be a positive integer")
        if length > STUDENT_MAX_SAMPLE_LENGTH:
            raise ValueError(
                f"sample length is capped at {STUDENT_MAX_SAMPLE_LENGTH} characters in student mode"
            )

        prompt = " ".join(prompt_args)
        clean_prompt, notes = sanitize_prompt(prompt, self.trainer.corpus)
        for note in notes:
            print(f"Note: {note}")
        print(
            self.trainer.model.generate(
                self.trainer.corpus,
                prompt=clean_prompt,
                length=length,
            )
        )

    def reset_to_fresh(self) -> None:
        checkpoint_path = self.checkpoint_path.expanduser().resolve()
        managed_checkpoint_path = student_checkpoint_path()
        if checkpoint_path != managed_checkpoint_path:
            raise ValueError("Student reset can only clear the managed local checkpoint file.")

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
        print("Reset complete. You are back to a fresh random model with no saved progress.")

    def print_doctor_report(self) -> None:
        for line in doctor_report(
            self.data_path,
            self.checkpoint_path,
            self.starter_checkpoint_path,
        ):
            print(line)

    @staticmethod
    def print_help() -> None:
        print("Commands:")
        print("  help                     Show this message")
        print("  sample [len] [prompt]    Generate text from the current model")
        print("  train [steps]            Train a little more and autosave")
        print(
            f"                           Student mode caps: sample <= {STUDENT_MAX_SAMPLE_LENGTH},"
            f" train <= {STUDENT_MAX_TRAIN_STEPS}"
        )
        print("  status                   Estimate current train/validation loss")
        print("  config                   Show the current settings")
        print("  reset                    Go back to a fresh untrained model")
        print("  doctor                   Check whether the package looks ready to use")
        print("  quit                     Exit the shell")


class TrainingShell:
    def __init__(
        self,
        trainer: TextGeneratorTrainer,
        data_path: Path,
        checkpoint_path: Path,
        allow_unsafe_checkpoint: bool = False,
    ) -> None:
        self.trainer = trainer
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.allow_unsafe_checkpoint = allow_unsafe_checkpoint

    def cmdloop(self) -> None:
        print("Interactive text-generator shell")
        print("Type 'help' for commands.")

        while True:
            try:
                line = input("textgen> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue

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

        try:
            if command in {"help", "?"}:
                self.print_help()
            elif command == "config":
                print(self.trainer.config())
            elif command == "status":
                print(self.trainer.status())
            elif command == "doctor":
                for line in doctor_report(
                    self.data_path,
                    self.checkpoint_path,
                    resolve_starter_checkpoint_path(),
                ):
                    print(line)
            elif command == "train":
                steps = int(args[0]) if args else 500
                self.trainer.train(steps=steps)
                self.trainer.save(self.checkpoint_path, self.data_path)
            elif command == "sample":
                length = 300
                prompt_args = args
                if args:
                    try:
                        length = int(args[0])
                        prompt_args = args[1:]
                    except ValueError:
                        prompt_args = args
                prompt = " ".join(prompt_args)
                clean_prompt, notes = sanitize_prompt(prompt, self.trainer.corpus)
                for note in notes:
                    print(f"Note: {note}")
                print(
                    self.trainer.model.generate(
                        self.trainer.corpus,
                        prompt=clean_prompt,
                        length=length,
                    )
                )
            elif command == "save":
                path = Path(args[0]) if args else self.checkpoint_path
                self.trainer.save(path, self.data_path)
            elif command == "load":
                allow_unsafe_checkpoint = self.allow_unsafe_checkpoint
                path_args: list[str] = []
                for arg in args:
                    if arg == "--unsafe":
                        allow_unsafe_checkpoint = True
                    else:
                        path_args.append(arg)
                if len(path_args) > 1:
                    raise ValueError("Usage: load [path] [--unsafe]")
                path = Path(path_args[0]) if path_args else self.checkpoint_path
                self.trainer = TextGeneratorTrainer.load(
                    path,
                    self.trainer.corpus,
                    self.data_path,
                    allow_unsafe_checkpoint=allow_unsafe_checkpoint,
                )
                print(f"Loaded checkpoint from {path}")
            elif command in {"rebuild", "rebuild-model", "rebuild_model"}:
                if len(args) not in {3, 5}:
                    print(
                        "Usage: rebuild <context_size> <hidden_dim> <learning_rate> "
                        "[batch_size] [embed_dim]"
                    )
                else:
                    self.rebuild_model(
                        context_size=int(args[0]),
                        hidden_dim=int(args[1]),
                        learning_rate=float(args[2]),
                        batch_size=int(args[3]) if len(args) == 5 else None,
                        embed_dim=int(args[4]) if len(args) == 5 else None,
                    )
            elif command in {"clear-models", "clear_models"}:
                self.clear_saved_models()
            elif command in {"quit", "exit"}:
                return False
            else:
                print(f"Unknown command: {command}")
        except Exception as exc:  # noqa: BLE001
            print(f"Command failed: {exc}")

        return True

    def clear_saved_models(self) -> None:
        checkpoint_path = self.checkpoint_path.expanduser().resolve()
        managed_checkpoint_dir = DEFAULT_CHECKPOINT_PATH.expanduser().resolve().parent
        protected_paths = {
            path
            for path in starter_checkpoint_candidates()
            if path.exists()
        }

        if checkpoint_path.is_relative_to(managed_checkpoint_dir):
            if not managed_checkpoint_dir.exists():
                print(f"No checkpoint directory found at {managed_checkpoint_dir}")
                return

            model_paths = sorted(
                path
                for path in managed_checkpoint_dir.rglob("*")
                if path.is_file() and is_checkpoint_file(path)
                and path.resolve() not in protected_paths
            )
            if not model_paths:
                print(f"No removable saved models found in {managed_checkpoint_dir}")
                return

            for path in model_paths:
                path.unlink()

            print(f"Deleted {len(model_paths)} saved model(s) from {managed_checkpoint_dir}")
            return

        if not checkpoint_path.exists():
            print(f"No saved model found at {checkpoint_path}")
            return

        if not checkpoint_path.is_relative_to(managed_checkpoint_dir):
            raise ValueError("Refusing to delete checkpoints outside the managed checkpoints directory.")
        if checkpoint_path in protected_paths:
            raise ValueError("Refusing to delete the bundled starter checkpoint.")

        checkpoint_path.unlink()
        print(f"Deleted checkpoint {checkpoint_path}")

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
        print("  doctor                   Check whether the package looks ready to use")
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
        print("                           No hard max is enforced, but larger context/hidden/batch/embed")
        print("                           values use more memory and train slower; very large lr can destabilize training")
        print("  clear-models             Delete saved progress checkpoints in checkpoints/")
        print("                           The bundled starter checkpoint is preserved.")
        print("  quit                     Exit the shell")


def build_trainer_from_args(args: argparse.Namespace) -> TextGeneratorTrainer:
    data_path = Path(args.data).expanduser().resolve()
    if args.command == "sample":
        if not args.fresh:
            if args.checkpoint is None:
                raise FileNotFoundError(
                    "No checkpoint specified for sample. "
                    "Pass --checkpoint to sample a saved model, or --fresh to sample from a new random model."
                )
            checkpoint_path = resolve_checkpoint_argument(args.checkpoint, DEFAULT_CHECKPOINT_PATH)
            if checkpoint_path.exists():
                return load_trainer_from_checkpoint(
                    checkpoint_path,
                    data_path=data_path,
                    context_size=args.context_size,
                    allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
                )
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}. "
                "Train a model first, or pass --fresh to sample from a new random model."
            )

    checkpoint_path = resolve_checkpoint_argument(args.checkpoint, DEFAULT_CHECKPOINT_PATH)
    if checkpoint_path.exists() and not args.fresh:
        return load_trainer_from_checkpoint(
            checkpoint_path,
            data_path=data_path,
            context_size=args.context_size,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
        )

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


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple character-level text generator trained on shakespeare.txt"
    )
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    common.add_argument("--checkpoint", default=None)
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

    student_parser = subparsers.add_parser(
        "student",
        parents=[common],
        help="Student-friendly interactive shell",
    )
    student_parser.set_defaults(command="student")

    shell_parser = subparsers.add_parser("shell", parents=[common], help="Teacher/advanced interactive shell")
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
    argv = sys.argv[1:]
    if not argv:
        argv = ["student"]
    args = parser.parse_args(argv)

    data_path = Path(args.data).expanduser().resolve()
    checkpoint_path = resolve_checkpoint_argument(args.checkpoint, DEFAULT_CHECKPOINT_PATH)

    if args.command == "doctor":
        for line in doctor_report(
            data_path,
            checkpoint_path,
            resolve_starter_checkpoint_path(),
        ):
            print(line)
        return 0

    if args.command == "student":
        try:
            trainer, source_checkpoint_path = build_student_trainer(args)
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(2, f"{parser.prog}: error: {exc}\n")
        shell = StudentShell(
            trainer=trainer,
            data_path=data_path,
            checkpoint_path=checkpoint_path,
            starter_checkpoint_path=resolve_starter_checkpoint_path(),
            source_checkpoint_path=source_checkpoint_path,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
        )
        shell.cmdloop()
        return 0

    try:
        trainer = build_trainer_from_args(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    if args.command == "shell":
        shell = TrainingShell(
            trainer,
            data_path=data_path,
            checkpoint_path=checkpoint_path,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
        )
        shell.cmdloop()
        return 0

    if args.command == "train":
        trainer.train(
            steps=args.steps,
            log_every=args.log_every,
        )
        trainer.save(checkpoint_path, data_path)
        print(trainer.status())
        return 0

    if args.command == "sample":
        prompt, notes = sanitize_prompt(args.prompt, trainer.corpus)
        for note in notes:
            print(f"Note: {note}")
        sample = trainer.model.generate(
            trainer.corpus,
            prompt=prompt,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(sample)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

import io
import pickle
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import zipfile
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import text_generator

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def make_trainer_for_text(text: str) -> text_generator.TextGeneratorTrainer:
    corpus = text_generator.Corpus(text=text, context_size=3)
    model = text_generator.CharMLP(
        vocab_size=corpus.vocab_size,
        context_size=corpus.context_size,
        embed_dim=4,
        hidden_dim=8,
        seed=7,
    )
    return text_generator.TextGeneratorTrainer(
        corpus=corpus,
        model=model,
        batch_size=2,
        learning_rate=1e-3,
        grad_clip=1.0,
        seed=7,
    )


def make_trainer() -> text_generator.TextGeneratorTrainer:
    return make_trainer_for_text("abbaabba\n")


def write_legacy_pickle_checkpoint(
    path: Path,
    data_path: Path,
    trainer: text_generator.TextGeneratorTrainer,
    include_corpus_hash: bool,
) -> None:
    payload = {
        "data_path": str(data_path),
        "chars": trainer.corpus.chars,
        "trainer": {
            "batch_size": trainer.batch_size,
            "learning_rate": trainer.learning_rate,
            "grad_clip": trainer.grad_clip,
            "seed": trainer.seed,
            "train_steps": trainer.train_steps,
            "rng_state": trainer.rng.bit_generator.state,
        },
        "model": trainer.model.to_checkpoint(),
    }
    if include_corpus_hash:
        payload["corpus_sha256"] = text_generator.corpus_fingerprint(trainer.corpus.text)

    with path.open("wb") as handle:
        pickle.dump(payload, handle)


@contextmanager
def patched_paths(root: Path):
    checkpoints_dir = root / "checkpoints"
    legacy_checkpoint_path = checkpoints_dir / "latest_model.npz"
    managed_sessions_dir = checkpoints_dir / "sessions"
    starter_checkpoint_path = checkpoints_dir / "starter_model.npz"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    with (
        mock.patch.object(text_generator, "CHECKPOINTS_DIR", checkpoints_dir),
        mock.patch.object(text_generator, "LEGACY_CHECKPOINT_PATH", legacy_checkpoint_path),
        mock.patch.object(text_generator, "MANAGED_SESSIONS_DIR", managed_sessions_dir),
        mock.patch.object(text_generator, "DEFAULT_STARTER_CHECKPOINT_PATH", starter_checkpoint_path),
    ):
        yield checkpoints_dir, legacy_checkpoint_path, managed_sessions_dir, starter_checkpoint_path


def run_main(argv: list[str], *, input_values: list[str] | None = None) -> tuple[int | None, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    input_patch = (
        mock.patch("builtins.input", side_effect=input_values)
        if input_values is not None
        else nullcontext()
    )
    with (
        mock.patch.object(sys, "argv", ["text_generator.py", *argv]),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
        input_patch,
    ):
        try:
            code = text_generator.main()
        except SystemExit as exc:
            code = exc.code
    return code, stdout.getvalue(), stderr.getvalue()


def spawn_lock_holder(lock_path: Path, label: str) -> subprocess.Popen[str]:
    script = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path

        sys.path.insert(0, sys.argv[1])
        import text_generator

        lock = text_generator.acquire_exclusive_lock(Path(sys.argv[2]), sys.argv[3])
        print("ready", flush=True)
        time.sleep(5)
        lock.release()
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(WORKSPACE_ROOT), str(lock_path), label],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    if line != "ready":
        stderr = process.stderr.read() if process.stderr is not None else ""
        process.kill()
        raise AssertionError(f"Lock holder failed to start: {line!r} stderr={stderr!r}")
    return process


def terminate_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    if process.stdout is not None:
        process.stdout.close()
    if process.stderr is not None:
        process.stderr.close()


class LogEveryValidationTests(unittest.TestCase):
    def test_train_rejects_zero_log_every(self) -> None:
        trainer = make_trainer()

        with self.assertRaisesRegex(ValueError, "log_every must be a positive integer"):
            trainer.train(steps=1, log_every=0)

    def test_train_rejects_negative_log_every(self) -> None:
        trainer = make_trainer()

        with self.assertRaisesRegex(ValueError, "log_every must be a positive integer"):
            trainer.train(steps=1, log_every=-5)

    def test_parser_rejects_zero_log_every(self) -> None:
        parser = text_generator.make_parser()

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                parser.parse_args(["train", "--log-every", "0"])

        self.assertEqual(exc.exception.code, 2)

    def test_parser_accepts_positive_log_every(self) -> None:
        parser = text_generator.make_parser()

        args = parser.parse_args(["train", "--log-every", "5"])

        self.assertEqual(args.log_every, 5)

    def test_train_logs_recent_losses_from_current_call(self) -> None:
        trainer = make_trainer()
        trainer.train_steps = 1
        x = np.zeros((trainer.batch_size, trainer.corpus.context_size), dtype=np.int32)
        y = np.zeros(trainer.batch_size, dtype=np.int32)
        losses_and_norms = [(1.0, 0.5), (2.0, 0.6), (3.0, 0.7)]

        with (
            mock.patch.object(trainer, "batch", return_value=(x, y)),
            mock.patch.object(trainer.model, "train_step", side_effect=losses_and_norms),
            mock.patch.object(trainer, "estimate_loss", return_value=9.0) as estimate_loss,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            trainer.train(steps=3, log_every=2)

        self.assertEqual(
            stdout.getvalue().strip().splitlines(),
            [
                "step       2 | train_loss=1.0000 | val_loss=9.0000 | grad_norm=0.5000",
                "step       4 | train_loss=2.5000 | val_loss=9.0000 | grad_norm=0.7000",
            ],
        )
        self.assertEqual(
            estimate_loss.call_args_list,
            [mock.call("val", num_batches=4), mock.call("val", num_batches=4)],
        )


class ApplyGradsTests(unittest.TestCase):
    def test_apply_grads_matches_reference_adam_update(self) -> None:
        model = text_generator.CharMLP(vocab_size=3, context_size=2, embed_dim=2, hidden_dim=2, seed=7)
        model.params = {
            "E": np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]], dtype=np.float32),
            "W1": np.array(
                [[0.2, -0.1], [0.0, 0.3], [-0.4, 0.5], [0.6, -0.7]], dtype=np.float32
            ),
            "b1": np.array([0.05, -0.15], dtype=np.float32),
            "W2": np.array([[0.25, -0.35, 0.45], [-0.55, 0.65, -0.75]], dtype=np.float32),
            "b2": np.array([0.01, -0.02, 0.03], dtype=np.float32),
        }
        model.opt_state = {
            "step": 0,
            "m": {name: np.zeros_like(value) for name, value in model.params.items()},
            "v": {name: np.zeros_like(value) for name, value in model.params.items()},
        }
        grads = {
            "E": np.array([[0.02, -0.01], [0.03, 0.04], [-0.05, 0.06]], dtype=np.float32),
            "W1": np.array(
                [[0.01, -0.02], [0.03, 0.04], [-0.05, 0.06], [0.07, -0.08]], dtype=np.float32
            ),
            "b1": np.array([0.02, -0.03], dtype=np.float32),
            "W2": np.array([[0.01, -0.02, 0.03], [-0.04, 0.05, -0.06]], dtype=np.float32),
            "b2": np.array([0.01, -0.02, 0.03], dtype=np.float32),
        }
        initial_params = {name: value.copy() for name, value in model.params.items()}
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        learning_rate = 1e-2
        grad_clip = 10.0

        expected_m = {name: np.zeros_like(value) for name, value in grads.items()}
        expected_v = {name: np.zeros_like(value) for name, value in grads.items()}
        expected_params = {name: value.copy() for name, value in initial_params.items()}
        expected_norm = float(np.sqrt(sum(np.sum(grad**2) for grad in grads.values())))
        step = 1

        for name, grad in grads.items():
            expected_m[name] = beta1 * expected_m[name] + (1.0 - beta1) * grad
            expected_v[name] = beta2 * expected_v[name] + (1.0 - beta2) * (grad * grad)
            m_hat = expected_m[name] / (1.0 - beta1**step)
            v_hat = expected_v[name] / (1.0 - beta2**step)
            expected_params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        observed_norm = model.apply_grads(
            {name: value.copy() for name, value in grads.items()},
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            grad_clip=grad_clip,
        )

        self.assertAlmostEqual(observed_norm, expected_norm)
        self.assertEqual(model.opt_state["step"], step)
        for name in expected_params:
            np.testing.assert_allclose(model.params[name], expected_params[name], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(
                model.opt_state["m"][name], expected_m[name], rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                model.opt_state["v"][name], expected_v[name], rtol=1e-6, atol=1e-6
            )


class PromptSanitizationTests(unittest.TestCase):
    def test_sanitize_prompt_normalizes_quotes_and_drops_unknown_chars(self) -> None:
        corpus = text_generator.Corpus(text="ROMEO:\nabc", context_size=3)

        cleaned, notes = text_generator.sanitize_prompt('“ROMEO:”🙂', corpus)

        self.assertEqual(cleaned, "ROMEO:")
        self.assertTrue(any("Normalized punctuation" in note for note in notes))
        self.assertTrue(any("Removed unsupported characters" in note for note in notes))

    def test_sanitize_prompt_reports_when_everything_is_removed(self) -> None:
        corpus = text_generator.Corpus(text="abc", context_size=2)

        cleaned, notes = text_generator.sanitize_prompt("🙂", corpus)

        self.assertEqual(cleaned, "")
        self.assertTrue(any("ended up empty" in note for note in notes))


class SessionManagerTests(unittest.TestCase):
    def test_target_for_session_uses_session_scoped_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()

                target = manager.target_for_session("alpha")

                self.assertTrue(target.managed)
                self.assertEqual(target.session_name, "alpha")
                self.assertEqual(
                    target.checkpoint_path,
                    (root / "checkpoints" / "sessions" / "alpha" / "latest_model.npz").resolve(),
                )
                self.assertEqual(
                    target.lock_path,
                    (root / "checkpoints" / "sessions" / "alpha" / "session.lock").resolve(),
                )

    def test_validate_name_rejects_invalid_session_names(self) -> None:
        manager = text_generator.SessionManager()

        for invalid_name in ("", "bad/name", "bad name", "../oops", "bad:idea"):
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaises(text_generator.SessionError):
                    manager.validate_name(invalid_name)

    def test_migrate_legacy_checkpoint_moves_legacy_into_default_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()
                data_path = root / "data.txt"
                data_path.write_text("abcaabca\n", encoding="utf-8")
                trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
                trainer.save(text_generator.LEGACY_CHECKPOINT_PATH, data_path)

                messages = manager.migrate_legacy_checkpoint()

                migrated_path = manager.checkpoint_path(text_generator.DEFAULT_SESSION_NAME)
                self.assertFalse(text_generator.LEGACY_CHECKPOINT_PATH.exists())
                self.assertTrue(migrated_path.exists())
                self.assertTrue(any("Migrated legacy saved progress" in message for message in messages))

    def test_migrate_legacy_checkpoint_warns_when_default_session_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()
                data_path = root / "data.txt"
                data_path.write_text("abcaabca\n", encoding="utf-8")
                trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
                trainer.save(text_generator.LEGACY_CHECKPOINT_PATH, data_path)
                trainer.save(manager.checkpoint_path(text_generator.DEFAULT_SESSION_NAME), data_path)

                messages = manager.migrate_legacy_checkpoint()

                self.assertTrue(text_generator.LEGACY_CHECKPOINT_PATH.exists())
                self.assertTrue(any("Warning:" in message for message in messages))

    def test_migrate_legacy_checkpoint_is_noop_without_legacy_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()

                messages = manager.migrate_legacy_checkpoint()

                self.assertEqual(messages, [])

    def test_choose_session_interactively_picks_existing_session_by_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()
                manager.ensure_session_dir("alpha")
                manager.ensure_session_dir("beta")

                with mock.patch("builtins.input", side_effect=["2"]):
                    chosen = manager.choose_session_interactively()

                self.assertEqual(chosen, "beta")

    def test_choose_session_interactively_creates_new_session_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()

                with mock.patch("builtins.input", side_effect=["alpha"]):
                    chosen = manager.choose_session_interactively()

                self.assertEqual(chosen, "alpha")
                self.assertTrue(manager.session_dir("alpha").exists())

    def test_choose_session_interactively_allows_numeric_session_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()
                manager.ensure_session_dir("alpha")

                with mock.patch("builtins.input", side_effect=["2"]):
                    chosen = manager.choose_session_interactively()

                self.assertEqual(chosen, "2")
                self.assertTrue(manager.session_dir("2").exists())

    def test_choose_session_interactively_repompts_on_invalid_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()

                with mock.patch("builtins.input", side_effect=["bad/name", "good-name"]):
                    with redirect_stdout(io.StringIO()) as stdout:
                        chosen = manager.choose_session_interactively()

                self.assertEqual(chosen, "good-name")
                self.assertIn("Session names may use only", stdout.getvalue())

    def test_choose_session_interactively_repompts_on_locked_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()
                manager.ensure_session_dir("alpha")
                process = spawn_lock_holder(manager.lock_path("alpha"), "Session 'alpha'")
                self.addCleanup(terminate_process, process)

                with mock.patch("builtins.input", side_effect=["1", "beta"]):
                    with redirect_stdout(io.StringIO()) as stdout:
                        chosen = manager.choose_session_interactively()

                self.assertEqual(chosen, "beta")
                self.assertIn("already in use", stdout.getvalue())


class CheckpointSafetyTests(unittest.TestCase):
    def test_save_writes_safe_archive_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            checkpoint_path = root / "model.npz"
            data_path.write_text("abcaabca\n", encoding="utf-8")

            trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            trainer.save(checkpoint_path, data_path)

            self.assertTrue(zipfile.is_zipfile(checkpoint_path))

    def test_save_uses_uncompressed_npz_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            checkpoint_path = root / "model.npz"
            data_path.write_text("abcaabca\n", encoding="utf-8")

            trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            trainer.save(checkpoint_path, data_path)

            with zipfile.ZipFile(checkpoint_path) as archive:
                self.assertTrue(archive.infolist())
                self.assertTrue(
                    all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
                )

    def test_safe_checkpoint_round_trips_after_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            checkpoint_path = root / "model.npz"
            data_path.write_text("abcaabca\n", encoding="utf-8")

            trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            trainer.train_steps = 12
            trainer.save(checkpoint_path, data_path)

            corpus = text_generator.Corpus(
                text=data_path.read_text(encoding="utf-8"),
                context_size=trainer.corpus.context_size,
            )
            loaded = text_generator.TextGeneratorTrainer.load(checkpoint_path, corpus)

            self.assertEqual(loaded.train_steps, 12)
            self.assertEqual(loaded.batch_size, trainer.batch_size)
            self.assertEqual(loaded.learning_rate, trainer.learning_rate)
            for name, param in trainer.model.params.items():
                np.testing.assert_allclose(loaded.model.params[name], param)

    def test_checkpoint_load_rejects_changed_corpus_with_same_charset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_data = root / "original.txt"
            changed_data = root / "changed.txt"
            checkpoint_path = root / "model.pkl"

            original_data.write_text("abccba\n", encoding="utf-8")
            changed_data.write_text("cbabac\n", encoding="utf-8")

            original_corpus = text_generator.Corpus(
                text=original_data.read_text(encoding="utf-8"),
                context_size=3,
            )
            model = text_generator.CharMLP(
                vocab_size=original_corpus.vocab_size,
                context_size=original_corpus.context_size,
                embed_dim=4,
                hidden_dim=8,
                seed=7,
            )
            trainer = text_generator.TextGeneratorTrainer(
                corpus=original_corpus,
                model=model,
                batch_size=2,
                learning_rate=1e-3,
                grad_clip=1.0,
                seed=7,
            )
            trainer.save(checkpoint_path, original_data)

            changed_corpus = text_generator.Corpus(
                text=changed_data.read_text(encoding="utf-8"),
                context_size=3,
            )

            with self.assertRaises(ValueError):
                text_generator.TextGeneratorTrainer.load(checkpoint_path, changed_corpus)

    def test_legacy_pickle_requires_explicit_unsafe_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            checkpoint_path = root / "legacy.pkl"
            data_path.write_text("abcaabca\n", encoding="utf-8")

            trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            write_legacy_pickle_checkpoint(
                checkpoint_path,
                data_path,
                trainer,
                include_corpus_hash=True,
            )

            corpus = text_generator.Corpus(
                text=data_path.read_text(encoding="utf-8"),
                context_size=3,
            )

            with self.assertRaisesRegex(ValueError, "Refusing to load a legacy pickle checkpoint"):
                text_generator.TextGeneratorTrainer.load(checkpoint_path, corpus)

            loaded = text_generator.TextGeneratorTrainer.load(
                checkpoint_path,
                corpus,
                allow_unsafe_checkpoint=True,
            )
            self.assertIsInstance(loaded, text_generator.TextGeneratorTrainer)

    def test_hashless_legacy_pickle_is_rejected_even_with_unsafe_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            checkpoint_path = root / "legacy.pkl"
            data_path.write_text("abccba\n", encoding="utf-8")

            trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            write_legacy_pickle_checkpoint(
                checkpoint_path,
                data_path,
                trainer,
                include_corpus_hash=False,
            )

            data_path.write_text("cbabac\n", encoding="utf-8")
            changed_corpus = text_generator.Corpus(
                text=data_path.read_text(encoding="utf-8"),
                context_size=3,
            )

            with self.assertRaisesRegex(ValueError, "missing a corpus fingerprint"):
                text_generator.TextGeneratorTrainer.load(
                    checkpoint_path,
                    changed_corpus,
                    allow_unsafe_checkpoint=True,
                )


class EmbeddingGradientAccumulationTests(unittest.TestCase):
    def test_embedding_grad_accumulation_matches_add_at_reference(self) -> None:
        trainer = make_trainer_for_text("abbaabba\n")
        x, y = trainer.batch("train")
        logits, cache = trainer.model.forward(x)
        probs = trainer.model._softmax(logits)
        batch_size = len(y)
        dlogits = probs
        dlogits[np.arange(batch_size), y] -= 1.0
        dlogits /= batch_size
        dhidden = dlogits @ trainer.model.params["W2"].T
        dhidden_linear = dhidden * (1.0 - cache["hidden"] ** 2)
        dflat = dhidden_linear @ trainer.model.params["W1"].T
        dembedding = dflat.reshape(batch_size, trainer.model.context_size, trainer.model.embed_dim)

        reference = np.zeros_like(trainer.model.params["E"])
        np.add.at(reference, cache["x"], dembedding)
        accumulated = trainer.model._accumulate_embedding_grads(cache["x"], dembedding)

        np.testing.assert_allclose(accumulated, reference, rtol=1e-6, atol=1e-6)


class LockingTests(unittest.TestCase):
    def test_same_session_lock_conflicts_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()
                manager.ensure_session_dir("alpha")
                process = spawn_lock_holder(manager.lock_path("alpha"), "Session 'alpha'")
                self.addCleanup(terminate_process, process)

                with self.assertRaises(text_generator.SessionLockError):
                    text_generator.acquire_exclusive_lock(
                        manager.lock_path("alpha"),
                        "Session 'alpha'",
                    )

    def test_different_session_locks_can_be_held_simultaneously(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager = text_generator.SessionManager()
                manager.ensure_session_dir("alpha")
                manager.ensure_session_dir("beta")
                process = spawn_lock_holder(manager.lock_path("alpha"), "Session 'alpha'")
                self.addCleanup(terminate_process, process)

                lock = text_generator.acquire_exclusive_lock(
                    manager.lock_path("beta"),
                    "Session 'beta'",
                )
                lock.release()

    def test_explicit_checkpoint_lock_conflicts_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / "model.npz"
            lock_path = text_generator.explicit_checkpoint_lock_path(checkpoint_path)
            process = spawn_lock_holder(lock_path, f"Checkpoint {checkpoint_path}")
            self.addCleanup(terminate_process, process)

            with self.assertRaises(text_generator.SessionLockError):
                text_generator.acquire_exclusive_lock(lock_path, f"Checkpoint {checkpoint_path}")


class InteractiveShellTests(unittest.TestCase):
    def make_shell(
        self,
        *,
        root: Path,
        managed: bool,
        session_name: str = "alpha",
    ) -> tuple[text_generator.SessionManager, text_generator.InteractiveShell, Path, text_generator.RunTarget]:
        data_path = root / "data.txt"
        data_path.write_text("abcaabca\n", encoding="utf-8")
        manager = text_generator.SessionManager()
        if managed:
            target = manager.target_for_session(session_name)
        else:
            target = manager.target_for_checkpoint(root / "manual.npz")
        trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
        shell = text_generator.InteractiveShell(
            trainer=trainer,
            data_path=data_path,
            run_target=target,
            session_manager=manager,
        )
        return manager, shell, data_path, target

    def test_interactive_shell_dispatches_help_status_config_session_and_quit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                _, shell, _, _ = self.make_shell(root=root, managed=True)

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("help"))
                self.assertIn("delete-session <name>", stdout.getvalue())
                self.assertIn("--context-size", stdout.getvalue())

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("status"))
                self.assertIn("steps=", stdout.getvalue())

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("config"))
                self.assertIn("seed=", stdout.getvalue())

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("session"))
                self.assertIn("Current session: alpha", stdout.getvalue())

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertFalse(shell.execute("quit"))
                self.assertEqual(stdout.getvalue(), "")

    def test_configure_shell_readline_uses_editline_binding(self) -> None:
        fake_readline = types.SimpleNamespace(backend="editline", parse_and_bind=mock.Mock())

        with mock.patch.dict(sys.modules, {"readline": fake_readline}):
            self.assertTrue(text_generator.configure_shell_readline())

        fake_readline.parse_and_bind.assert_called_once_with("bind ^L ed-clear-screen")

    def test_configure_shell_readline_uses_gnu_readline_binding(self) -> None:
        fake_readline = types.SimpleNamespace(backend="readline", parse_and_bind=mock.Mock())

        with mock.patch.dict(sys.modules, {"readline": fake_readline}):
            self.assertTrue(text_generator.configure_shell_readline())

        fake_readline.parse_and_bind.assert_called_once_with('"\\C-l": clear-screen')

    def test_cmdloop_clears_screen_when_form_feed_line_is_entered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                _, shell, _, _ = self.make_shell(root=root, managed=True)

                with (
                    mock.patch.object(text_generator, "configure_shell_readline", return_value=False),
                    mock.patch.object(text_generator, "clear_terminal_screen") as clear_terminal_screen,
                    mock.patch("builtins.input", side_effect=["\x0c", "quit"]),
                    redirect_stdout(io.StringIO()),
                ):
                    shell.cmdloop()

                clear_terminal_screen.assert_called_once_with()

    def test_rebuild_named_args_update_only_requested_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                _, shell, _, _ = self.make_shell(root=root, managed=True)

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("rebuild -c=5 -e=6"))

                self.assertIn("context=5", stdout.getvalue())
                self.assertEqual(shell.trainer.corpus.context_size, 5)
                self.assertEqual(shell.trainer.model.embed_dim, 6)
                self.assertEqual(shell.trainer.model.hidden_dim, 8)
                self.assertEqual(shell.trainer.batch_size, 2)
                self.assertEqual(shell.trainer.learning_rate, 1e-3)

    def test_rebuild_named_args_support_separate_option_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                _, shell, _, _ = self.make_shell(root=root, managed=True)

                with redirect_stdout(io.StringIO()):
                    self.assertTrue(shell.execute("rebuild --learning-rate 0.02 -b 4"))

                self.assertEqual(shell.trainer.corpus.context_size, 3)
                self.assertEqual(shell.trainer.model.hidden_dim, 8)
                self.assertEqual(shell.trainer.model.embed_dim, 4)
                self.assertEqual(shell.trainer.batch_size, 4)
                self.assertEqual(shell.trainer.learning_rate, 0.02)

    def test_rebuild_positional_form_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                _, shell, _, _ = self.make_shell(root=root, managed=True)

                with redirect_stdout(io.StringIO()):
                    self.assertTrue(shell.execute("rebuild 6 10 0.02"))

                self.assertEqual(shell.trainer.corpus.context_size, 6)
                self.assertEqual(shell.trainer.model.hidden_dim, 10)
                self.assertEqual(shell.trainer.learning_rate, 0.02)
                self.assertEqual(shell.trainer.batch_size, 2)
                self.assertEqual(shell.trainer.model.embed_dim, 4)

    def test_rebuild_rejects_mixed_positional_and_named_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                _, shell, _, _ = self.make_shell(root=root, managed=True)

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("rebuild 6 10 0.02 -e=5"))

                self.assertIn("Do not mix positional values", stdout.getvalue())
                self.assertEqual(shell.trainer.corpus.context_size, 3)
                self.assertEqual(shell.trainer.model.embed_dim, 4)

    def test_rebuild_rejects_unknown_named_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                _, shell, _, _ = self.make_shell(root=root, managed=True)

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("rebuild -x=5"))

                self.assertIn("Unknown rebuild option", stdout.getvalue())
                self.assertEqual(shell.trainer.corpus.context_size, 3)

    def test_sessions_command_marks_current_and_locked_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager, shell, _, _ = self.make_shell(root=root, managed=True, session_name="alpha")
                manager.ensure_session_dir("beta")
                process = spawn_lock_holder(manager.lock_path("beta"), "Session 'beta'")
                self.addCleanup(terminate_process, process)

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("sessions"))

                output = stdout.getvalue()
                self.assertIn("alpha (current, locked)", output)
                self.assertIn("beta (locked)", output)

    def test_reset_clears_only_current_session_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager, shell, data_path, target = self.make_shell(root=root, managed=True, session_name="alpha")
                trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
                trainer.save(target.checkpoint_path, data_path)
                beta_target = manager.target_for_session("beta")
                trainer.save(beta_target.checkpoint_path, data_path)

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("reset"))

                self.assertIn("Reset complete", stdout.getvalue())
                self.assertFalse(target.checkpoint_path.exists())
                self.assertTrue(beta_target.checkpoint_path.exists())

    def test_reset_is_rejected_for_explicit_checkpoint_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                _, shell, _, _ = self.make_shell(root=root, managed=False)

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("reset"))

                self.assertIn("reset is only available for managed sessions", stdout.getvalue())

    def test_delete_session_refuses_current_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                _, shell, _, _ = self.make_shell(root=root, managed=True, session_name="alpha")

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("delete-session alpha"))

                self.assertIn("Refusing to delete the current session", stdout.getvalue())

    def test_delete_session_refuses_locked_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patched_paths(root):
                manager, shell, _, _ = self.make_shell(root=root, managed=True, session_name="alpha")
                manager.ensure_session_dir("beta")
                process = spawn_lock_holder(manager.lock_path("beta"), "Session 'beta'")
                self.addCleanup(terminate_process, process)

                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertTrue(shell.execute("delete-session beta"))

                self.assertIn("already in use", stdout.getvalue())


class CommandScopeTests(unittest.TestCase):
    def test_train_command_requires_session_or_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            data_path.write_text("abcaabca\n", encoding="utf-8")
            with patched_paths(root):
                code, _, stderr = run_main(["train", "--data", str(data_path)])

                self.assertEqual(code, 2)
                self.assertIn("train requires --session or --checkpoint", stderr)

    def test_sample_command_requires_session_checkpoint_or_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            data_path.write_text("abcaabca\n", encoding="utf-8")
            with patched_paths(root):
                code, _, stderr = run_main(["sample", "--data", str(data_path)])

                self.assertEqual(code, 2)
                self.assertIn("sample requires --session, --checkpoint, or --fresh", stderr)

    def test_train_command_with_session_saves_session_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            data_path.write_text("abcaabca\n", encoding="utf-8")
            with patched_paths(root):
                code, stdout, stderr = run_main(
                    [
                        "train",
                        "--data",
                        str(data_path),
                        "--session",
                        "alpha",
                        "--context-size",
                        "3",
                        "--embed-dim",
                        "4",
                        "--hidden-dim",
                        "8",
                        "--batch-size",
                        "2",
                        "--learning-rate",
                        "0.001",
                        "--steps",
                        "1",
                        "--log-every",
                        "1",
                    ]
                )

                manager = text_generator.SessionManager()
                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                self.assertTrue(manager.checkpoint_path("alpha").exists())
                self.assertIn("Saved checkpoint", stdout)

    def test_sample_command_with_session_reads_session_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            data_path.write_text("abcaabca\n", encoding="utf-8")
            with patched_paths(root):
                manager = text_generator.SessionManager()
                target = manager.target_for_session("alpha")
                trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
                trainer.save(target.checkpoint_path, data_path)

                code, stdout, stderr = run_main(
                    [
                        "sample",
                        "--data",
                        str(data_path),
                        "--session",
                        "alpha",
                        "--context-size",
                        "3",
                        "--length",
                        "5",
                    ]
                )

                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                self.assertTrue(stdout.strip())

    def test_shell_command_with_session_uses_named_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            data_path.write_text("abcaabca\n", encoding="utf-8")
            with patched_paths(root):
                code, stdout, stderr = run_main(
                    [
                        "shell",
                        "--data",
                        str(data_path),
                        "--session",
                        "alpha",
                        "--context-size",
                        "3",
                        "--embed-dim",
                        "4",
                        "--hidden-dim",
                        "8",
                        "--batch-size",
                        "2",
                        "--learning-rate",
                        "0.001",
                    ],
                    input_values=["quit"],
                )

                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                self.assertIn("Current session: alpha", stdout)

    def test_main_defaults_to_shell_and_prompts_for_session_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            data_path.write_text("abcaabca\n", encoding="utf-8")
            with patched_paths(root):
                code, stdout, stderr = run_main(
                    [
                        "--data",
                        str(data_path),
                        "--context-size",
                        "3",
                        "--embed-dim",
                        "4",
                        "--hidden-dim",
                        "8",
                        "--batch-size",
                        "2",
                        "--learning-rate",
                        "0.001",
                    ],
                    input_values=["alpha", "quit"],
                )

                manager = text_generator.SessionManager()
                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                self.assertTrue(manager.session_dir("alpha").exists())
                self.assertIn("Managed sessions:", stdout)


class LauncherSetupTests(unittest.TestCase):
    def test_launchers_stay_self_contained_and_do_not_auto_install(self) -> None:
        root = Path(__file__).resolve().parents[1]
        shared_launcher = (root / "start_shell.command").read_text(encoding="utf-8")
        student_launcher = (root / "start_student.command").read_text(encoding="utf-8")
        teacher_launcher = (root / "start_teacher.command").read_text(encoding="utf-8")

        self.assertNotIn("../.venv/bin/python", shared_launcher)
        self.assertNotIn("../shakespeare/.venv/bin/python", shared_launcher)
        self.assertIn("Install now? [y/N]", shared_launcher)

        for launcher in (student_launcher, teacher_launcher):
            self.assertIn('exec /bin/zsh "$SCRIPT_DIR/start_shell.command"', launcher)

        self.assertNotIn("Teacher mode includes advanced commands", teacher_launcher)
        self.assertNotIn("Continue? [y/N]", teacher_launcher)

    def test_requirements_are_pinned_for_reproducible_installs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        numpy_lines = [line.strip() for line in requirements if line.strip() and not line.startswith("#")]

        self.assertEqual(len(numpy_lines), 1)
        self.assertTrue(numpy_lines[0].startswith("numpy=="))


if __name__ == "__main__":
    unittest.main()

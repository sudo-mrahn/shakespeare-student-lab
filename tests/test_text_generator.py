import pickle
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import text_generator


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

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["train", "--log-every", "0"])

        self.assertEqual(exc.exception.code, 2)

    def test_parser_accepts_positive_log_every(self) -> None:
        parser = text_generator.make_parser()

        args = parser.parse_args(["train", "--log-every", "5"])

        self.assertEqual(args.log_every, 5)


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

    def test_sample_requires_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.txt"
            checkpoint_path = Path(tmpdir) / "missing.pkl"
            data_path.write_text("abcaabca\n", encoding="utf-8")

            parser = text_generator.make_parser()
            args = parser.parse_args(
                [
                    "sample",
                    "--data",
                    str(data_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--length",
                    "5",
                ]
            )

            with self.assertRaises(FileNotFoundError):
                text_generator.build_trainer_from_args(args)

    def test_clear_models_preserves_unrelated_pickles_outside_checkpoint_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            checkpoint_path = root / "model.pkl"
            unrelated_dir = root / "nested"
            unrelated_dir.mkdir()

            data_path.write_text("abcaabca\n", encoding="utf-8")
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            unrelated_pickle = unrelated_dir / "other.pkl"
            unrelated_pickle.write_text("keep me", encoding="utf-8")

            trainer = make_trainer()
            shell = text_generator.TrainingShell(
                trainer=trainer,
                data_path=data_path,
                checkpoint_path=checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "outside the managed checkpoints directory"):
                shell.clear_saved_models()

            self.assertTrue(unrelated_pickle.exists())

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
                text_generator.TextGeneratorTrainer.load(checkpoint_path, corpus, data_path)

            loaded = text_generator.TextGeneratorTrainer.load(
                checkpoint_path,
                corpus,
                data_path,
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
                    data_path,
                    allow_unsafe_checkpoint=True,
                )


class LauncherSetupTests(unittest.TestCase):
    def test_launchers_stay_self_contained_and_do_not_auto_install(self) -> None:
        root = Path(__file__).resolve().parents[1]
        student_launcher = (root / "start_student.command").read_text(encoding="utf-8")
        teacher_launcher = (root / "start_teacher.command").read_text(encoding="utf-8")

        for launcher in (student_launcher, teacher_launcher):
            self.assertNotIn("../.venv/bin/python", launcher)
            self.assertNotIn("../shakespeare/.venv/bin/python", launcher)
            self.assertIn("Install now? [y/N]", launcher)

        self.assertIn("Continue? [y/N]", teacher_launcher)

    def test_requirements_are_pinned_for_reproducible_installs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        numpy_lines = [line.strip() for line in requirements if line.strip() and not line.startswith("#")]

        self.assertEqual(len(numpy_lines), 1)
        self.assertTrue(numpy_lines[0].startswith("numpy=="))


class StudentFlowTests(unittest.TestCase):
    def test_build_student_trainer_starts_fresh_when_latest_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            latest_path = root / "latest_model.npz"
            starter_path = root / "starter_model.npz"
            data_path.write_text("abcaabca\n", encoding="utf-8")

            trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            trainer.save(starter_path, data_path)

            old_starter = text_generator.DEFAULT_STARTER_CHECKPOINT_PATH
            text_generator.DEFAULT_STARTER_CHECKPOINT_PATH = starter_path
            try:
                parser = text_generator.make_parser()
                args = parser.parse_args(
                    [
                        "student",
                        "--data",
                        str(data_path),
                        "--checkpoint",
                        str(latest_path),
                        "--context-size",
                        "3",
                    ]
                )

                loaded_trainer, source_path = text_generator.build_student_trainer(args)
            finally:
                text_generator.DEFAULT_STARTER_CHECKPOINT_PATH = old_starter

            self.assertIsInstance(loaded_trainer, text_generator.TextGeneratorTrainer)
            self.assertIsNone(source_path)
            self.assertEqual(loaded_trainer.train_steps, 0)

    def test_sample_does_not_silently_fall_back_to_the_starter_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            starter_path = root / "starter_model.npz"
            data_path.write_text("abcaabca\n", encoding="utf-8")

            starter_trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            starter_trainer.train_steps = 17
            starter_trainer.save(starter_path, data_path)

            old_starter = text_generator.DEFAULT_STARTER_CHECKPOINT_PATH
            text_generator.DEFAULT_STARTER_CHECKPOINT_PATH = starter_path
            try:
                parser = text_generator.make_parser()
                args = parser.parse_args(
                    [
                        "sample",
                        "--data",
                        str(data_path),
                        "--length",
                        "5",
                    ]
                )
                trainer = text_generator.build_trainer_from_args(args)
            except FileNotFoundError:
                return
            finally:
                text_generator.DEFAULT_STARTER_CHECKPOINT_PATH = old_starter

            self.assertEqual(trainer.train_steps, 0)
            self.assertFalse(
                np.array_equal(
                    trainer.model.params["E"],
                    starter_trainer.model.params["E"],
                )
            )

    def test_student_package_no_longer_ships_a_bundled_legacy_checkpoint(self) -> None:
        self.assertFalse(text_generator.CHECKPOINTS_DIR.joinpath("default_100k.pkl").exists())

    def test_student_shell_reset_restores_fresh_model_and_clears_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            latest_path = root / "latest_model.npz"
            starter_path = root / "starter_model.npz"
            data_path.write_text("abcaabca\n", encoding="utf-8")

            starter_trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            starter_trainer.train_steps = 123
            starter_trainer.save(starter_path, data_path)

            fresh_trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            old_checkpoint = text_generator.DEFAULT_CHECKPOINT_PATH
            text_generator.DEFAULT_CHECKPOINT_PATH = latest_path
            try:
                shell = text_generator.StudentShell(
                    trainer=fresh_trainer,
                    data_path=data_path,
                    checkpoint_path=latest_path,
                    starter_checkpoint_path=starter_path,
                    source_checkpoint_path=None,
                )

                shell.reset_to_fresh()
            finally:
                text_generator.DEFAULT_CHECKPOINT_PATH = old_checkpoint

            self.assertFalse(latest_path.exists())
            self.assertEqual(shell.trainer.train_steps, 0)
            self.assertIsNone(shell.source_checkpoint_path)

    def test_student_shell_reset_does_not_delete_external_checkpoint_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            external_checkpoint = root / "outside.pkl"
            data_path.write_text("abcaabca\n", encoding="utf-8")
            external_checkpoint.write_text("keep me", encoding="utf-8")

            trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            shell = text_generator.StudentShell(
                trainer=trainer,
                data_path=data_path,
                checkpoint_path=external_checkpoint,
                starter_checkpoint_path=None,
                source_checkpoint_path=None,
            )

            with self.assertRaisesRegex(ValueError, "managed local checkpoint file"):
                shell.reset_to_fresh()

            self.assertTrue(external_checkpoint.exists())

    def test_student_shell_rejects_oversized_train_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.txt"
            latest_path = root / "latest_model.npz"
            data_path.write_text("abcaabca\n", encoding="utf-8")

            trainer = make_trainer_for_text(data_path.read_text(encoding="utf-8"))
            shell = text_generator.StudentShell(
                trainer=trainer,
                data_path=data_path,
                checkpoint_path=latest_path,
                starter_checkpoint_path=None,
                source_checkpoint_path=None,
            )

            with self.assertRaisesRegex(ValueError, "capped at"):
                shell.train([str(text_generator.STUDENT_MAX_TRAIN_STEPS + 1)])


if __name__ == "__main__":
    unittest.main()

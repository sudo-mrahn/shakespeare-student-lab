# Shakespeare Student Lab

Shakespeare Student Lab is a student-friendly text generation sandbox built for classroom exploration.
Students start from a fresh random model, alternate between `sample` and `train`, and watch the model slowly learn the style of the Shakespeare corpus.

This package is designed to be simple enough for high school students to use without needing the full research or engineering workflow.

## What It Includes

- a default student shell with a very small command set
- a separate teacher shell for advanced commands
- safe checkpoint handling based on `.npz` files
- a fresh-model student flow with a safe `reset`
- built-in guardrails on sample length and training steps

## Platform Notes

The one-click launchers in this repo are for macOS.

Students should have:

- `python3` available
- internet access on first launch so NumPy can install into a local `.venv`

## Quick Start

For students on Mac:

```bash
./start_student.command
```

That opens the student shell, which starts from a fresh random model unless the student has already saved progress in `checkpoints/latest_model.npz`.

Suggested first commands:

```text
sample 500
train 100
status
```

## Student Commands

Inside the student shell, the main commands are:

```text
sample [length] [prompt]
train [steps]
status
reset
doctor
help
quit
```

Student mode guardrails:

- sample length is capped at `1000`
- training is capped at `5000` steps at a time
- unsupported prompt characters are cleaned up with a note instead of crashing
- `reset` returns to a fresh untrained model and clears saved progress safely

The first outputs will usually look like nonsense. That is expected and part of the learning experience.

## Teacher Mode

For the advanced shell:

```bash
./start_teacher.command
```

Teacher mode keeps the fuller workflow, including commands such as `save`, `load`, `rebuild`, and `clear-models`.
The launcher adds an extra confirmation step before opening that mode.

## Classroom Handoff

If you are sending this to students as a zip:

1. Remove `.venv` if it exists.
2. Zip the folder.
3. Tell students to double-click `start_student.command`.

The first launch creates a local `.venv` and may prompt to install requirements from `requirements.txt`.

## Safety Notes

- normal checkpoints use the safer `.npz` format
- legacy pickle checkpoints are blocked unless explicitly allowed with `--allow-unsafe-checkpoint`
- student reset only clears managed student progress
- teacher/demo starter checkpoints are protected from accidental deletion

## Files

- `text_generator.py`: main program
- `shakespeare.txt`: training corpus
- `checkpoints/starter_model.npz`: optional bundled starter checkpoint for teacher/demo use
- `checkpoints/latest_model.npz`: saved student progress after training
- `start_student.command`: student launcher
- `start_teacher.command`: teacher launcher

## Public Domain Text

The Shakespeare source text in this repo is public domain.

## License

This project is released under the MIT License. See `LICENSE`.

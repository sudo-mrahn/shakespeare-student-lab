# Shakespeare Text Lab

This version of the project is packaged for classroom use.
Students start from a fresh, untrained model, train a little at a time, and reset safely without touching the advanced model-building commands.

## Start Here

On Mac, double-click `start_student.command`.

If you prefer the terminal:

```bash
./start_student.command
```

That opens the student shell by default.

## Teacher Note

Before sending this folder to students, remove `.venv` if it exists and then zip the folder.
Students should use a Mac with `python3` available and internet access the first time they launch the package.

## Student Commands

Inside the student shell, the main commands are:

```text
sample 250 "ROMEO:"
train 200
status
reset
help
quit
```

What they do:

- `sample [length] [prompt]` generates text from the current model.
- `train [steps]` trains a little more and automatically saves progress.
- `status` shows the current loss estimates.
- `reset` goes back to a fresh untrained model and clears saved progress.
- `doctor` checks whether the package looks ready to use.

Student mode has guardrails:

- sample length is capped at 1000 characters
- training is capped at 5000 steps at a time
- unsupported prompt characters are cleaned up with a note instead of crashing the session
- the first output will usually look like nonsense until students train for a while

## Files

- `text_generator.py`: the program
- `shakespeare.txt`: the training text
- `checkpoints/latest_model.npz`: each student's saved progress
- `start_student.command`: one-click student launcher for Mac
- `start_teacher.command`: one-click advanced shell for Mac

## Teacher Mode

If you want the original advanced workflow, run:

```bash
./start_teacher.command
```

Or double-click `start_teacher.command`.

Teacher mode still supports commands like `save`, `load`, `rebuild`, and `clear-models`.
An optional starter checkpoint is still included for teacher/demo use and is protected from accidental deletion.
The teacher launcher asks before starting advanced mode and before installing requirements.

## Setup

If Python packages are not installed yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The launcher scripts create a local `.venv` and ask before installing requirements on the first run.
If NumPy is not already installed in that local environment, the first launch needs network access to fetch it.

## Safety Notes

- Current checkpoints use a safe NumPy archive format (`.npz`) for normal saves and loads.
- Legacy pickle checkpoints are blocked unless explicitly allowed with `--allow-unsafe-checkpoint`.
- `reset` is safer than manually deleting checkpoint files.

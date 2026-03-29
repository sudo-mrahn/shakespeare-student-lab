# Shakespeare Text Lab

This project is a small character-level Shakespeare text generator packaged as an interactive shell.
It now uses named sessions so multiple instances can run on the same VM without fighting over one shared checkpoint file.

## Start Here

On Mac, double-click either launcher:

```bash
./start_student.command
```

or

```bash
./start_teacher.command
```

Both compatibility launchers now start the same unified shell.
They both forward into the shared `start_shell.command` launcher.

If you launch the shell without `--session`, it will list existing sessions and ask you to choose one or create a new one.

## Named Sessions

Managed progress is stored under:

```text
checkpoints/sessions/<session_name>/latest_model.npz
```

This means different sessions can train and sample at the same time on the same machine.
If two processes try to use the same session at once, the second one fails fast with a clear lock message instead of corrupting the checkpoint.

The first time you run the new version, an old legacy checkpoint at `checkpoints/latest_model.npz` is automatically migrated into session `default`.

## Interactive Shell

Running the script with no subcommand opens the shell:

```bash
python3 text_generator.py
```

Useful shell commands:

```text
help
status
config
doctor
session
sessions
train 200
sample 250 "ROMEO:"
rebuild -c=100 -e=50
reset
delete-session old-run
quit
```

Notes:

- `rebuild` accepts named options like `-c=100 -e=50`; unspecified values keep their current settings.
- The older positional form `rebuild <context_size> <hidden_dim> <learning_rate> [batch_size] [embed_dim]` still works.
- `reset` only clears the current managed session.
- `delete-session <name>` refuses the current session and any locked session.
- `save` and `load` still support manual checkpoint paths for advanced workflows.

## Command Line Usage

Use `--session` for managed state:

```bash
python3 text_generator.py shell --session romeo
python3 text_generator.py train --session romeo --steps 500
python3 text_generator.py sample --session romeo --length 300
```

Use `--checkpoint` for an explicit manual checkpoint path:

```bash
python3 text_generator.py shell --checkpoint /tmp/demo.npz
python3 text_generator.py train --checkpoint /tmp/demo.npz --steps 500
python3 text_generator.py sample --checkpoint /tmp/demo.npz --length 300
```

Use `--fresh` when you want a new random model instead of loading saved state:

```bash
python3 text_generator.py sample --fresh --length 200
```

Non-interactive `train` requires `--session` or `--checkpoint` so it knows where to save progress.
Non-interactive `sample` requires `--session`, `--checkpoint`, or `--fresh`.

## Files

- `text_generator.py`: the program
- `shakespeare.txt`: the training text
- `checkpoints/sessions/`: managed session directories
- `checkpoints/starter_model.npz`: optional manual starter checkpoint
- `start_shell.command`: shared launcher used by the compatibility wrappers
- `start_student.command`: compatibility launcher for the unified shell
- `start_teacher.command`: compatibility launcher for the unified shell

## Setup

If Python packages are not installed yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The launcher scripts create a local `.venv` when possible and ask before installing requirements on the first run.
If the folder path contains `:`, they automatically store the virtual environment under `~/.shakespeare-text-lab/venvs/...` because macOS Python refuses to create a venv inside colon-containing paths.
If you are setting up manually from a folder whose path contains `:`, create the virtual environment somewhere outside the project folder instead of using `.venv`.
If NumPy is not already installed in that local environment, the first launch needs network access to fetch it.

## Safety Notes

- Current checkpoints use a safe NumPy archive format (`.npz`) for normal saves and loads.
- Legacy pickle checkpoints are blocked unless explicitly allowed with `--allow-unsafe-checkpoint`.
- Managed sessions are locked so concurrent writers fail fast instead of silently overwriting each other.

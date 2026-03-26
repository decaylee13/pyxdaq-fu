# xdaq_rl scaffold

Direct Python RL pipeline over `pyxdaq` with modular boundaries:

- `hardware/` backend wrappers (`real` and `sim`)
- `buffer/` ring buffer
- `runtime/` acquisition service
- `features/` observation extraction
- `env/` Gym environment
- `training/` runnable entrypoint
- `logging/` replay datasets

## Quick start

```bash
source .venv/bin/activate
python -m xdaq_rl.training.run_random --mode sim --episodes 1 --steps 200
```

Real hardware mode:

```bash
python -m xdaq_rl.training.run_random --mode real --device-index 0 --rhs
```

Use `--log-dir /tmp/xdaq-rl-replay` to save episode `.npz` files.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from xdaq_rl.config import EnvConfig, HardwareConfig
from xdaq_rl.env.xdaq_env import XDAQEnv
from xdaq_rl.features.basic import RMSFeatureExtractor
from xdaq_rl.logging.replay_writer import ReplayWriter
from xdaq_rl.runtime.acquisition import AcquisitionService
from xdaq_rl.runtime.factory import create_backend, create_ring


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Random-policy smoke run for XDAQ Gym env")
    p.add_argument("--mode", choices=["sim", "real"], default="sim")
    p.add_argument("--rhs", action="store_true")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--sample-rate", type=int, default=30000)
    p.add_argument("--streams", type=int, default=2)
    p.add_argument("--chunk-samples", type=int, default=128)
    p.add_argument("--window-size", type=int, default=1024)
    p.add_argument("--step-horizon", type=int, default=128)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--log-dir", type=Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    hcfg = HardwareConfig(
        mode=args.mode,
        rhs=args.rhs,
        device_index=args.device_index,
        sample_rate_hz=args.sample_rate,
        num_streams=args.streams,
        chunk_samples=args.chunk_samples,
        stream_kind="AC" if args.rhs else "continuous",
    )
    ecfg = EnvConfig(window_size=args.window_size, step_horizon=args.step_horizon, max_steps=args.steps)

    backend = create_backend(hcfg)
    ring = create_ring(hcfg)
    acq = AcquisitionService(backend=backend, ring=ring)

    extractor = RMSFeatureExtractor()
    env = XDAQEnv(acquisition=acq, ring=ring, feature_extractor=extractor, cfg=ecfg)

    replay = ReplayWriter(args.log_dir) if args.log_dir is not None else None

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            for _ in range(args.steps):
                action = env.action_space.sample().astype(np.float32)
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = bool(terminated or truncated)
                ep_reward += reward
                if replay is not None:
                    replay.append(obs=obs, action=float(action[0]), reward=reward, done=done)
                if done:
                    break

            out_msg = f"episode={ep} reward={ep_reward:.3f} latest_index={step_info['latest_sample_index']}"
            if replay is not None:
                path = replay.flush(episode_id=ep)
                out_msg += f" replay={path}"
            print(out_msg)
    finally:
        env.close()


if __name__ == "__main__":
    main()

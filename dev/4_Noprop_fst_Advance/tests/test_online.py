from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.online import OnlineConfig, OnlineRunner, RewardSequenceEnvironment


def make_runner() -> OnlineRunner:
    sequences = np.array(
        [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]], dtype=np.int8
    )
    config = OnlineConfig(10, 0.5, 4, 4, 1, 0.1, 0.1)
    return OnlineRunner(RewardSequenceEnvironment(sequences), config, "two_block_mse", 12345)


def test_online_order_information_and_real_buffer() -> None:
    runner = make_runner()
    runner.run()
    assert runner.actions[:2] == [0, 1]
    assert len(runner.buffer_rewards) == 10
    assert runner.synthetic_buffer_insertions == 0
    assert not hasattr(runner, "means")
    assert not hasattr(runner.environment, "means")
    assert runner.block_1.update_count == runner.block_2.update_count == 10
    for round_index in range(10):
        events = runner.event_log[4 * round_index : 4 * (round_index + 1)]
        assert [event["event"] for event in events] == [
            "decision",
            "observation",
            "buffer_append",
            "update",
        ]


def test_checkpoint_resume_is_exact(tmp_path: Path) -> None:
    full = make_runner()
    full.run()
    partial = make_runner()
    partial.run(until=6)
    checkpoint = tmp_path / "runner.pkl"
    partial.save_checkpoint(checkpoint)
    resumed = OnlineRunner.from_checkpoint(checkpoint)
    resumed.run()
    assert full.actions == resumed.actions
    assert full.rewards == resumed.rewards
    assert full.losses == resumed.losses
    assert np.array_equal(full.block_1.parameters, resumed.block_1.parameters)
    assert np.array_equal(full.block_2.parameters, resumed.block_2.parameters)


def test_balanced_buffer_does_not_invent_missing_arm() -> None:
    sequences = np.ones((2, 4), dtype=np.int8)
    config = OnlineConfig(4, 0.5, 2, 4, 1, 0.1, 0.1, balanced_buffer=True)
    runner = OnlineRunner(RewardSequenceEnvironment(sequences), config, "two_block_mse", 77)
    runner.step_once()
    assert runner.buffer_actions == [0]
    runner.step_once()
    assert runner.buffer_actions == [0, 1]
    assert runner.synthetic_buffer_insertions == 0

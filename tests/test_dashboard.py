from __future__ import annotations

import unittest

from typesafe_mario.actions import JUMP_RELEASE_ACTION, Action
from typesafe_mario.dashboard import DashboardCommand, clamp01


class DashboardTests(unittest.TestCase):
    def test_clamp01_handles_missing_and_out_of_range_values(self) -> None:
        self.assertEqual(clamp01(None), 0.0)
        self.assertEqual(clamp01(-0.2), 0.0)
        self.assertEqual(clamp01(0.42), 0.42)
        self.assertEqual(clamp01(1.8), 1.0)

    def test_jump_macros_have_non_jump_release_frames(self) -> None:
        self.assertEqual(JUMP_RELEASE_ACTION[Action.RIGHT_JUMP], Action.RIGHT)
        self.assertEqual(JUMP_RELEASE_ACTION[Action.RIGHT_RUN_JUMP], Action.RIGHT_RUN)
        self.assertEqual(JUMP_RELEASE_ACTION[Action.JUMP], Action.NOOP)

    def test_dashboard_commands_are_stable_strings(self) -> None:
        self.assertEqual(DashboardCommand.RESTART, "restart")
        self.assertEqual(DashboardCommand.QUIT, "quit")


if __name__ == "__main__":
    unittest.main()

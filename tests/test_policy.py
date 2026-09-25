from __future__ import annotations

import unittest

from typesafe_mario.actions import Action
from typesafe_mario.policy import _wall_climb_step
from typesafe_mario.state import MarioStateParser


def _wall_snapshot(parser: MarioStateParser, goomba_x: int | None = None):
    """Grounded Mario at x=100 facing a four-tile wall one tile ahead."""
    ram = bytearray(0x0800)
    for column in range(16):
        ram[0x0500 + 5 * 16 + column] = 1  # ground row
    for row in range(1, 5):
        ram[0x0500 + row * 16 + 7] = 1  # four-tile wall, one tile right of Mario
    if goomba_x is not None:
        ram[0x000F] = 1
        ram[0x0016] = 0x06
        ram[0x006E] = 0
        ram[0x0087] = goomba_x
        ram[0x00CF] = 80
    info = {
        "world": 1,
        "stage": 1,
        "area": 1,
        "x_pos": 100,
        "y_pos": 80,
        "y_pixel": 80,
        "progress": 100,
        "time": 390,
        "life": 2,
        "status": "small",
    }
    return parser.parse(info, ram)


class WallClimbTests(unittest.TestCase):
    def test_climb_maneuver_runs_without_enemy_pressure(self) -> None:
        snapshot = _wall_snapshot(MarioStateParser())

        climb, phase = _wall_climb_step(snapshot, None)

        self.assertEqual(climb, Action.LEFT)
        self.assertEqual(phase, "retreat")

    def test_climb_yields_when_enemy_is_adjacent(self) -> None:
        snapshot = _wall_snapshot(MarioStateParser(), goomba_x=116)

        climb, phase = _wall_climb_step(snapshot, None)

        self.assertIsNone(climb)
        self.assertIsNone(phase)

    def test_climb_yields_when_contact_is_imminent(self) -> None:
        parser = MarioStateParser()
        _wall_snapshot(parser, goomba_x=164)
        closing = _wall_snapshot(parser, goomba_x=148)  # 16 px/frame closing speed

        climb, phase = _wall_climb_step(closing, None)

        self.assertIsNone(climb)
        self.assertIsNone(phase)

    def test_takeoff_accounts_for_response_delay(self) -> None:
        # Charging at 2 px/frame with an 8-frame response delay covers a whole
        # tile before the button press lands, so a wall measured 4 tiles out is
        # already inside the 3-tile takeoff window when the action executes.
        wall_ram = bytearray(0x0800)
        for column in range(16):
            wall_ram[0x0500 + 5 * 16 + column] = 1  # ground
        for row in range(1, 5):
            wall_ram[0x0500 + row * 16 + 11] = 1  # four-tile wall at world x=176

        def moving_snapshot(parser: MarioStateParser, x: int, delay: int):
            info = {
                "world": 1,
                "stage": 1,
                "area": 1,
                "x_pos": x,
                "y_pos": 80,
                "y_pixel": 80,
                "progress": x,
                "time": 390,
                "life": 2,
                "status": "small",
            }
            return parser.parse(
                info,
                wall_ram,
                previous_response_delay_frames=delay,
                elapsed_frames=8,
            )

        no_delay = MarioStateParser()
        moving_snapshot(no_delay, 100, 0)
        snapshot = moving_snapshot(no_delay, 116, 0)
        self.assertEqual(_wall_climb_step(snapshot, None), (Action.RIGHT_RUN, "charge"))

        delayed = MarioStateParser()
        moving_snapshot(delayed, 100, 8)
        snapshot = moving_snapshot(delayed, 116, 8)
        self.assertEqual(_wall_climb_step(snapshot, None), (Action.RIGHT_RUN_JUMP, None))


if __name__ == "__main__":
    unittest.main()

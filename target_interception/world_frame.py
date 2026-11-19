"""Room world frame: the 4.4 x 3.6 m arena used as the out-of-distribution test.

The large training arena has its own frame (scenarios/large_arena.py). Scenario pools,
suites and checkpoints record the frame ID so the two can never be mixed.
"""

WORLD_FRAME_ID = "mockup_room_20260917"
WORLD_AXIS_CONVENTION = "X long-room/forward, Y short-room/left, Z up"
ARENA_MIN_M = (-2.20, -1.80, 0.05)
ARENA_MAX_M = (2.20, 1.80, 1.90)

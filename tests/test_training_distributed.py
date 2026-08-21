import unittest

from queroquero.training_distributed import (
    DistributedContext,
    global_step_batch,
    rank_step_batch,
)


class DistributedTrainingTests(unittest.TestCase):
    def test_two_rank_shards_are_disjoint_and_cover_the_global_batch(self) -> None:
        schedule = list(range(24))
        global_batch = global_step_batch(schedule, 2, 8)
        rank_zero = rank_step_batch(
            schedule,
            2,
            rank=0,
            world_size=2,
            micro_batch_size_per_rank=1,
            accumulation_steps_per_rank=4,
        )
        rank_one = rank_step_batch(
            schedule,
            2,
            rank=1,
            world_size=2,
            micro_batch_size_per_rank=1,
            accumulation_steps_per_rank=4,
        )

        self.assertEqual(rank_zero, [8, 9, 10, 11])
        self.assertEqual(rank_one, [12, 13, 14, 15])
        self.assertFalse(set(rank_zero) & set(rank_one))
        self.assertEqual(rank_zero + rank_one, global_batch)

    def test_resume_cursor_starts_at_the_next_global_batch(self) -> None:
        schedule = list(range(48))
        completed_steps = 3
        rank_zero = rank_step_batch(
            schedule,
            completed_steps + 1,
            rank=0,
            world_size=2,
            micro_batch_size_per_rank=1,
            accumulation_steps_per_rank=4,
        )
        rank_one = rank_step_batch(
            schedule,
            completed_steps + 1,
            rank=1,
            world_size=2,
            micro_batch_size_per_rank=1,
            accumulation_steps_per_rank=4,
        )

        self.assertEqual(rank_zero + rank_one, list(range(24, 32)))

    def test_uninitialized_context_reductions_are_identity_operations(self) -> None:
        context = DistributedContext(
            torch=None,
            strategy="single_process",
            backend="none",
            rank=0,
            local_rank=0,
            world_size=1,
            device="cuda:0",
            initialized=False,
        )

        self.assertEqual(context.reduce_sum(1.25), 1.25)
        self.assertEqual(context.reduce_max(2.5), 2.5)
        self.assertTrue(context.any_true(True))
        self.assertEqual(context.all_gather_objects("digest"), ["digest"])
        self.assertEqual(context.broadcast_object({"ok": True}), {"ok": True})

    def test_invalid_rank_or_short_schedule_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            rank_step_batch(
                list(range(8)),
                1,
                rank=2,
                world_size=2,
                micro_batch_size_per_rank=1,
                accumulation_steps_per_rank=4,
            )
        with self.assertRaisesRegex(RuntimeError, "ended"):
            global_step_batch(list(range(7)), 1, 8)


if __name__ == "__main__":
    unittest.main()

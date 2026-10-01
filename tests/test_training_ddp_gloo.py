import importlib.util
import random
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _gloo_worker(
    rank,
    world_size,
    init_path,
    output_path,
    checkpoint_path,
    resume,
    stop_step,
    arm,
):
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 4),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(4, 1),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0 - step / 4
        )
        start_step = 0
        random.seed(42 + rank)
        torch.manual_seed(42 + rank)
        if resume:
            state = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            rank_rng = state["rng_by_rank"][rank]
            random.setstate(rank_rng["python"])
            torch.set_rng_state(rank_rng["torch"])
            start_step = state["step"]

        ddp = torch.nn.parallel.DistributedDataParallel(
            model,
            static_graph=False,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        for step in range(start_step + 1, 5):
            optimizer.zero_grad(set_to_none=True)
            global_values = list(range((step - 1) * 8, step * 8))
            if arm == "general":
                global_values = [
                    value + 64 if value % 3 == 0 else value
                    for value in global_values
                ]
            local_values = global_values[rank * 4 : (rank + 1) * 4]
            for index, value in enumerate(local_values):
                sync = ddp.no_sync() if index < 3 else _NullContext()
                with sync:
                    x = torch.tensor([[value / 32.0]], dtype=torch.float32)
                    target = 0.5 * x + 0.25
                    prediction = ddp(x)
                    jitter = 0.99 + random.random() * 0.02
                    loss = torch.nn.functional.mse_loss(prediction, target)
                    (loss * jitter / 4).backward()
            optimizer.step()
            scheduler.step()

            if stop_step == step:
                local_rng = {
                    "rank": rank,
                    "python": random.getstate(),
                    "torch": torch.get_rng_state(),
                }
                rng_by_rank = [None for _ in range(world_size)]
                dist.all_gather_object(rng_by_rank, local_rng)
                if rank == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "rng_by_rank": rng_by_rank,
                            "step": step,
                            "cursor": step * 8,
                        },
                        checkpoint_path,
                    )
                dist.barrier()
                return

        if rank == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": 4,
                    "cursor": 32,
                },
                output_path,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the preparation venv")
class GlooResumeIntegrationTests(unittest.TestCase):
    def test_continuous_and_resumed_ddp_training_are_identical(self) -> None:
        import torch
        import torch.multiprocessing as multiprocessing

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for arm in ("general", "forum_tech"):
                with self.subTest(arm=arm):
                    continuous = root / f"{arm}-continuous.pt"
                    resumed = root / f"{arm}-resumed.pt"
                    checkpoint = root / f"{arm}-checkpoint.pt"
                    multiprocessing.spawn(
                        _gloo_worker,
                        args=(
                            2,
                            (root / f"{arm}-continuous-init").as_posix(),
                            continuous.as_posix(),
                            checkpoint.as_posix(),
                            False,
                            None,
                            arm,
                        ),
                        nprocs=2,
                        join=True,
                    )
                    multiprocessing.spawn(
                        _gloo_worker,
                        args=(
                            2,
                            (root / f"{arm}-stop-init").as_posix(),
                            resumed.as_posix(),
                            checkpoint.as_posix(),
                            False,
                            2,
                            arm,
                        ),
                        nprocs=2,
                        join=True,
                    )
                    checkpoint_state = torch.load(checkpoint, weights_only=False)
                    self.assertEqual(checkpoint_state["cursor"], 16)
                    self.assertEqual(len(checkpoint_state["rng_by_rank"]), 2)
                    multiprocessing.spawn(
                        _gloo_worker,
                        args=(
                            2,
                            (root / f"{arm}-resume-init").as_posix(),
                            resumed.as_posix(),
                            checkpoint.as_posix(),
                            True,
                            None,
                            arm,
                        ),
                        nprocs=2,
                        join=True,
                    )

                    continuous_state = torch.load(continuous, weights_only=False)
                    resumed_state = torch.load(resumed, weights_only=False)
                    self.assertEqual(
                        continuous_state["cursor"], resumed_state["cursor"]
                    )
                    self.assertEqual(
                        continuous_state["scheduler"], resumed_state["scheduler"]
                    )
                    for name, tensor in continuous_state["model"].items():
                        self.assertTrue(
                            torch.equal(tensor, resumed_state["model"][name])
                        )
                    self._assert_nested_equal(
                        continuous_state["optimizer"],
                        resumed_state["optimizer"],
                        torch,
                    )

    def _assert_nested_equal(self, left, right, torch) -> None:
        self.assertEqual(type(left), type(right))
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self._assert_nested_equal(left[key], right[key], torch)
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self._assert_nested_equal(left_item, right_item, torch)
        else:
            self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Sequence, TypeVar


T = TypeVar("T")


@dataclass
class DistributedContext:
    torch: Any
    strategy: str
    backend: str
    rank: int
    local_rank: int
    world_size: int
    device: Any
    initialized: bool

    @classmethod
    def initialize(
        cls, torch: Any, config: Dict[str, Any]
    ) -> "DistributedContext":
        execution = config["execution"]
        expected_world_size = execution["world_size"]
        strategy = execution["strategy"]
        if strategy == "single_process":
            actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if actual_world_size != 1:
                raise RuntimeError("single-process training requires WORLD_SIZE=1")
            if torch.cuda.device_count() != 1:
                raise RuntimeError("single-process training requires one visible GPU")
            torch.cuda.set_device(0)
            return cls(
                torch=torch,
                strategy=strategy,
                backend=execution["backend"],
                rank=0,
                local_rank=0,
                world_size=1,
                device=torch.device("cuda", 0),
                initialized=False,
            )

        required = {"RANK", "LOCAL_RANK", "WORLD_SIZE"}
        missing = sorted(required - set(os.environ))
        if missing:
            raise RuntimeError(
                "DDP training must be launched by torchrun; missing "
                + ", ".join(missing)
            )
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != expected_world_size:
            raise RuntimeError(
                f"DDP requires WORLD_SIZE={expected_world_size}; found {world_size}"
            )
        if torch.cuda.device_count() != expected_world_size:
            raise RuntimeError(
                f"DDP requires exactly {expected_world_size} visible GPUs"
            )
        if not 0 <= rank < world_size or not 0 <= local_rank < world_size:
            raise RuntimeError("DDP rank metadata is outside the expected range")
        if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
            raise RuntimeError("the PyTorch build must provide distributed NCCL support")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend=execution["backend"],
            init_method="env://",
            timeout=timedelta(minutes=10),
        )
        return cls(
            torch=torch,
            strategy=strategy,
            backend=execution["backend"],
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=torch.device("cuda", local_rank),
            initialized=True,
        )

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.initialized:
            self.torch.distributed.barrier()

    def close(self) -> None:
        if self.initialized and self.torch.distributed.is_initialized():
            self.torch.distributed.destroy_process_group()
            self.initialized = False

    def broadcast_object(self, value: T | None, *, source: int = 0) -> T:
        if not self.initialized:
            return value  # type: ignore[return-value]
        values = [value if self.rank == source else None]
        self.torch.distributed.broadcast_object_list(values, src=source)
        return values[0]

    def all_gather_objects(self, value: T) -> list[T]:
        if not self.initialized:
            return [value]
        values: list[T | None] = [None for _ in range(self.world_size)]
        self.torch.distributed.all_gather_object(values, value)
        return values  # type: ignore[return-value]

    def reduce_sum(self, value: float) -> float:
        if not self.initialized:
            return float(value)
        tensor = self.torch.tensor(
            float(value), dtype=self.torch.float64, device=self.device
        )
        self.torch.distributed.all_reduce(
            tensor, op=self.torch.distributed.ReduceOp.SUM
        )
        return float(tensor.item())

    def reduce_max(self, value: float) -> float:
        if not self.initialized:
            return float(value)
        tensor = self.torch.tensor(
            float(value), dtype=self.torch.float64, device=self.device
        )
        self.torch.distributed.all_reduce(
            tensor, op=self.torch.distributed.ReduceOp.MAX
        )
        return float(tensor.item())

    def any_true(self, value: bool) -> bool:
        if not self.initialized:
            return value
        tensor = self.torch.tensor(
            int(value), dtype=self.torch.int32, device=self.device
        )
        self.torch.distributed.all_reduce(
            tensor, op=self.torch.distributed.ReduceOp.MAX
        )
        return bool(tensor.item())


def global_step_batch(
    schedule: Sequence[T], optimizer_step: int, global_batch_size: int
) -> list[T]:
    if optimizer_step < 1:
        raise ValueError("optimizer_step must be positive")
    start = (optimizer_step - 1) * global_batch_size
    batch = list(schedule[start : start + global_batch_size])
    if len(batch) != global_batch_size:
        raise RuntimeError("global training schedule ended before the optimizer step")
    return batch


def rank_step_batch(
    schedule: Sequence[T],
    optimizer_step: int,
    *,
    rank: int,
    world_size: int,
    micro_batch_size_per_rank: int,
    accumulation_steps_per_rank: int,
) -> list[T]:
    local_batch_size = micro_batch_size_per_rank * accumulation_steps_per_rank
    global_batch_size = local_batch_size * world_size
    if not 0 <= rank < world_size:
        raise ValueError("rank must be within world_size")
    batch = global_step_batch(schedule, optimizer_step, global_batch_size)
    start = rank * local_batch_size
    return batch[start : start + local_batch_size]

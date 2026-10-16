"""Sparsity schedules: how much of the network should be gone at step t."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CubicSchedule:
    """Gradual pruning, Zhu & Gupta (2017):

        s(t) = s_f + (s_i - s_f) * (1 - (t - t0) / (t_end - t0))**3,   t0 <= t <= t_end

    with a prune event every ``frequency`` steps. The derivative is largest at
    t0 and zero at t_end: remove a lot early, while the over-parameterised
    network has plenty of redundancy, and progressively less as each remaining
    weight matters more, leaving recovery time between events.
    ``end_step`` is snapped down to the last prune event so the schedule ends
    exactly on an event (and therefore exactly at the target).
    """

    final_sparsity: float
    begin_step: int
    end_step: int
    frequency: int = 100
    initial_sparsity: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.initial_sparsity <= self.final_sparsity < 1.0:
            raise ValueError("need 0 <= initial_sparsity <= final_sparsity < 1")
        if self.end_step <= self.begin_step or self.frequency < 1:
            raise ValueError("need end_step > begin_step and frequency >= 1")
        snapped = (
            self.begin_step + (self.end_step - self.begin_step) // self.frequency * self.frequency
        )
        object.__setattr__(self, "end_step", snapped)

    def sparsity(self, step: int) -> float:
        if step < self.begin_step:
            return self.initial_sparsity if step >= 0 else 0.0
        if step >= self.end_step:
            return self.final_sparsity
        progress = (step - self.begin_step) / (self.end_step - self.begin_step)
        return (
            self.final_sparsity
            + (self.initial_sparsity - self.final_sparsity) * (1.0 - progress) ** 3
        )

    def is_prune_step(self, step: int) -> bool:
        return (
            self.begin_step <= step <= self.end_step
            and (step - self.begin_step) % self.frequency == 0
        )


@dataclass(frozen=True)
class OneShotSchedule:
    """Prune to the target in a single event at ``step`` (the weaker baseline)."""

    final_sparsity: float
    step: int

    def sparsity(self, step: int) -> float:
        return self.final_sparsity if step >= self.step else 0.0

    def is_prune_step(self, step: int) -> bool:
        return step == self.step

    @property
    def end_step(self) -> int:
        return self.step

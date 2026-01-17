from craptorch.core.training import CosineSchedule

import numpy as np

def test_unit_cosine_schedule():
    schedule = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)

    lr_start = schedule.get_lr(0)
    lr_middle = schedule.get_lr(50)
    lr_end = schedule.get_lr(100)

    assert abs(lr_start - 0.1) < 1e-6, f"Expected 0.1 at start, got {lr_start}"
    assert abs(lr_end - 0.01) < 1e-6, f"Expected 0.01 at end, got {lr_end}"
    assert 0.01 < lr_middle < 0.1, f"Middle LR should be between min and max, got {lr_middle}"

    # Test monotonic decrease in first half
    lr_quarter = schedule.get_lr(25)
    assert lr_quarter > lr_middle, "LR should decrease monotonically in first half"

if __name__ == "__main__":
    test_unit_cosine_schedule()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

from torch.utils.data import DataLoader, Dataset


class Wine2DDataset(Dataset):
    """Dataset that samples random transitions on a 4x4 grid."""

    GRID_SIZE = 4
    NUM_STATES = GRID_SIZE * GRID_SIZE
    NUM_ACTIONS = 4

    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        current_state = random.randrange(self.NUM_STATES)
        action = random.randrange(self.NUM_ACTIONS)
        next_state = self._transition(current_state, action)
        return current_state, action, next_state

    @classmethod
    def _transition(cls, state, action):
        row, col = divmod(state, cls.GRID_SIZE)

        if action == 0:  # up
            row = max(row - 1, 0)
        elif action == 1:  # down
            row = min(row + 1, cls.GRID_SIZE - 1)
        elif action == 2:  # left
            col = max(col - 1, 0)
        elif action == 3:  # right
            col = min(col + 1, cls.GRID_SIZE - 1)
        else:
            raise ValueError(f"Invalid action {action}. Expected one of 0, 1, 2, 3.")

        return row * cls.GRID_SIZE + col


if __name__ == "__main__":
    dataset = Wine2DDataset(num_samples=1024)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    current_state, action, next_state = next(iter(dataloader))
    print("current_state:", current_state)
    print("action:", action)
    print("next_state:", next_state)

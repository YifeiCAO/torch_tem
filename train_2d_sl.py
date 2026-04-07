#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Supervised next-state training for TEM on a 4x4 grid."""

import torch
from torch import nn
from torch.utils.data import DataLoader

import model
import parameters
from dataset_wine_2d import Wine2DDataset


class SupervisedTEMTransition(nn.Module):
    """Wrap the TEM transition model with a state encoder and classifier."""

    def __init__(self, tem_model, num_states=16):
        super().__init__()
        self.tem = tem_model
        self.num_states = num_states
        self.tem_g_sizes = list(self.tem.hyper["n_g"])
        self.tem_g_size = sum(self.tem_g_sizes)

        self.state_embedding = nn.Embedding(num_states, self.tem_g_size)
        self.classifier = nn.Linear(self.tem_g_size, num_states)

        nn.init.xavier_uniform_(self.state_embedding.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _split_g(self, g_flat):
        g_modules = []
        start = 0
        for module_size in self.tem_g_sizes:
            stop = start + module_size
            g_modules.append(g_flat[:, start:stop])
            start = stop
        return g_modules

    def forward(self, current_state_index, action_index):
        g_current_flat = self.state_embedding(current_state_index)
        g_current = self._split_g(g_current_flat)

        # TEM expects actions as a Python list of integer labels for the batch.
        action_list = action_index.detach().cpu().tolist()
        g_next = self.tem.f_mu_g_path(action_list, g_current)
        g_next_flat = torch.cat(g_next, dim=1)
        logits = self.classifier(g_next_flat)
        return logits, g_next_flat


def build_supervised_model(batch_size=64, num_states=16, device="cpu"):
    params = parameters.parameters()
    params["has_static_action"] = False
    params["batch_size"] = batch_size

    tem = model.Model(params).to(device)
    supervised_model = SupervisedTEMTransition(tem_model=tem, num_states=num_states).to(device)
    return supervised_model


def load_supervised_model(checkpoint_path, batch_size=64, num_states=16, device="cpu"):
    supervised_model = build_supervised_model(
        batch_size=batch_size,
        num_states=num_states,
        device=device,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    supervised_model.load_state_dict(state_dict)
    supervised_model.eval()
    return supervised_model


def train(
    num_epochs=20,
    num_samples=4096,
    batch_size=64,
    learning_rate=1e-3,
):
    torch.manual_seed(0)

    # Keep training on CPU: the original TEM implementation creates CPU tensors internally.
    device = torch.device("cpu")

    supervised_model = build_supervised_model(
        batch_size=batch_size,
        num_states=16,
        device=device,
    )

    dataset = Wine2DDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(supervised_model.parameters(), lr=learning_rate)

    supervised_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for current_state, action, next_state in dataloader:
            current_state = current_state.to(device=device, dtype=torch.long)
            action = action.to(device=device, dtype=torch.long)
            next_state = next_state.to(device=device, dtype=torch.long)

            optimizer.zero_grad()

            logits, _ = supervised_model(current_state, action)
            loss = criterion(logits, next_state)
            loss.backward()
            optimizer.step()

            batch_size_actual = current_state.size(0)
            epoch_loss += loss.item() * batch_size_actual
            epoch_total += batch_size_actual
            epoch_correct += (logits.argmax(dim=1) == next_state).sum().item()

        mean_loss = epoch_loss / epoch_total
        accuracy = 100.0 * epoch_correct / epoch_total
        print(
            f"Epoch {epoch + 1:03d}/{num_epochs:03d} "
            f"| loss={mean_loss:.4f} "
            f"| acc={accuracy:.2f}%"
        )

    return supervised_model


if __name__ == "__main__":
    train()

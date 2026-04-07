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

    INVERSE_ACTION = {
        0: 1,  # up -> down
        1: 0,  # down -> up
        2: 3,  # left -> right
        3: 2,  # right -> left
    }

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

    def encode_state(self, state_index):
        return self.state_embedding(state_index)

    def transition_g(self, g_flat, action_index):
        g_current = self._split_g(g_flat)
        action_list = action_index.detach().cpu().tolist()
        g_next = self.tem.f_mu_g_path(action_list, g_current)
        return torch.cat(g_next, dim=1)

    def inverse_action(self, action_index):
        inverse = [self.INVERSE_ACTION[int(action)] for action in action_index.detach().cpu().tolist()]
        return torch.tensor(inverse, device=action_index.device, dtype=action_index.dtype)

    def forward(self, current_state_index, action_index):
        g_current_flat = self.encode_state(current_state_index)
        g_next_flat = self.transition_g(g_current_flat, action_index)
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
    latent_consistency_weight=0.1,
    cycle_consistency_weight=0.1,
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

    classification_criterion = nn.CrossEntropyLoss()
    consistency_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(supervised_model.parameters(), lr=learning_rate)

    supervised_model.train()
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_latent_loss = 0.0
        epoch_cycle_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for current_state, action, next_state in dataloader:
            current_state = current_state.to(device=device, dtype=torch.long)
            action = action.to(device=device, dtype=torch.long)
            next_state = next_state.to(device=device, dtype=torch.long)

            optimizer.zero_grad()

            logits, g_next_pred = supervised_model(current_state, action)
            classification_loss = classification_criterion(logits, next_state)

            g_next_target = supervised_model.encode_state(next_state)
            latent_consistency_loss = consistency_criterion(
                g_next_pred,
                g_next_target,
            )

            inverse_action = supervised_model.inverse_action(action)
            cycle_reconstruction = supervised_model.transition_g(g_next_pred, inverse_action)
            cycle_consistency_loss = consistency_criterion(
                cycle_reconstruction,
                supervised_model.encode_state(current_state),
            )

            total_loss = (
                classification_loss
                + latent_consistency_weight * latent_consistency_loss
                + cycle_consistency_weight * cycle_consistency_loss
            )

            total_loss.backward()
            optimizer.step()

            batch_size_actual = current_state.size(0)
            epoch_total_loss += total_loss.item() * batch_size_actual
            epoch_classification_loss += classification_loss.item() * batch_size_actual
            epoch_latent_loss += latent_consistency_loss.item() * batch_size_actual
            epoch_cycle_loss += cycle_consistency_loss.item() * batch_size_actual
            epoch_total += batch_size_actual
            epoch_correct += (logits.argmax(dim=1) == next_state).sum().item()

        mean_total_loss = epoch_total_loss / epoch_total
        mean_classification_loss = epoch_classification_loss / epoch_total
        mean_latent_loss = epoch_latent_loss / epoch_total
        mean_cycle_loss = epoch_cycle_loss / epoch_total
        accuracy = 100.0 * epoch_correct / epoch_total
        print(
            f"Epoch {epoch + 1:03d}/{num_epochs:03d} "
            f"| total={mean_total_loss:.4f} "
            f"| ce={mean_classification_loss:.4f} "
            f"| latent={mean_latent_loss:.4f} "
            f"| cycle={mean_cycle_loss:.4f} "
            f"| acc={accuracy:.2f}%"
        )

    return supervised_model


if __name__ == "__main__":
    train()

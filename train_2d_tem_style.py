#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a 4x4 next-state task using a TEM-style sequential objective."""

import argparse
import random

import numpy as np
import torch
from torch import nn

import model
import parameters


GRID_SIZE = 4
NUM_STATES = GRID_SIZE * GRID_SIZE
NUM_ACTIONS = 4

ACTION_DELTAS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def transition(state, action):
    row, col = divmod(state, GRID_SIZE)
    delta_row, delta_col = ACTION_DELTAS[action]

    new_row = min(max(row + delta_row, 0), GRID_SIZE - 1)
    new_col = min(max(col + delta_col, 0), GRID_SIZE - 1)
    return new_row * GRID_SIZE + new_col


def sample_walk(rollout_length):
    states = [random.randrange(NUM_STATES)]
    actions = []
    for _ in range(rollout_length):
        action = random.randrange(NUM_ACTIONS)
        next_state = transition(states[-1], action)
        actions.append(action)
        states.append(next_state)
    return states, actions


def make_location(state_id):
    return {"id": state_id, "shiny": None}


def sample_observation_map(n_x):
    return np.random.permutation(n_x)[:NUM_STATES].tolist()


def make_observation(state_id, observation_map, n_x, device):
    observation_index = observation_map[state_id]
    observation = torch.zeros(n_x, dtype=torch.float, device=device)
    observation[observation_index] = 1.0
    return observation


def build_tem_batch(batch_size, rollout_length, n_x, device):
    observation_maps = [sample_observation_map(n_x) for _ in range(batch_size)]
    batch_walks = [sample_walk(rollout_length) for _ in range(batch_size)]

    chunk = []
    next_state_labels = []

    for step_index in range(rollout_length):
        locations = []
        observations = []
        actions = []
        next_states = []

        for env_index in range(batch_size):
            states, walk_actions = batch_walks[env_index]
            current_state = states[step_index]
            next_state = states[step_index + 1]
            action = walk_actions[step_index]

            locations.append(make_location(current_state))
            observations.append(
                make_observation(
                    current_state,
                    observation_maps[env_index],
                    n_x,
                    device,
                )
            )
            actions.append(action)
            next_states.append(next_state)

        chunk.append([locations, torch.stack(observations, dim=0), actions])
        next_state_labels.append(torch.tensor(next_states, device=device, dtype=torch.long))

    return chunk, next_state_labels


class TEMNextStatePredictor(nn.Module):
    def __init__(self, tem_model):
        super().__init__()
        self.tem = tem_model
        self.tem_g_size = sum(self.tem.hyper["n_g"])
        self.classifier = nn.Linear(self.tem_g_size, NUM_STATES)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def predict_next_logits(self, g_current, action_index):
        g_next = self.tem.f_mu_g_path(action_index, g_current)
        g_next_flat = torch.cat(g_next, dim=1)
        logits = self.classifier(g_next_flat)
        return logits, g_next_flat


def build_model(batch_size, device):
    params = parameters.parameters()
    params["has_static_action"] = False
    params["n_actions"] = NUM_ACTIONS
    params["batch_size"] = batch_size

    tem = model.Model(params).to(device)
    tem.hyper = move_to_device(tem.hyper, device)
    return TEMNextStatePredictor(tem).to(device), params


def load_model(checkpoint_path, batch_size, device):
    predictor, params = build_model(batch_size=batch_size, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.eval()
    return predictor, checkpoint.get("train_config", {}), params


def train(
    train_steps=2000,
    batch_size=16,
    rollout_length=20,
    supervised_weight=1.0,
    tem_weight=1.0,
    checkpoint_path="tem_2d_style.pt",
    checkpoint_every=250,
    seed=0,
    device=None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    predictor, params = build_model(batch_size=batch_size, device=device)
    tem = predictor.tem

    params["train_it"] = train_steps
    optimizer = torch.optim.Adam(predictor.parameters(), lr=params["lr_max"])
    supervised_criterion = nn.CrossEntropyLoss()

    for train_step in range(train_steps):
        eta_new, lambda_new, p2g_scale_offset, lr, _, loss_weights = parameters.parameter_iteration(train_step, params)
        tem.hyper["eta"] = eta_new
        tem.hyper["lambda"] = lambda_new
        tem.hyper["p2g_scale_offset"] = p2g_scale_offset
        loss_weights = loss_weights.to(device)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        chunk, next_state_labels = build_tem_batch(
            batch_size=batch_size,
            rollout_length=rollout_length,
            n_x=tem.hyper["n_x"],
            device=device,
        )

        optimizer.zero_grad()
        forward = tem(chunk, prev_iter=None)

        tem_loss = torch.tensor(0.0, device=device)
        supervised_loss = torch.tensor(0.0, device=device)
        accuracy_correct = 0
        accuracy_total = 0
        visited = [[False for _ in range(NUM_STATES)] for _ in range(batch_size)]

        for step_index, step in enumerate(forward):
            step_loss = []
            for env_index, env_visited in enumerate(visited):
                state_id = step.g[env_index]["id"]
                if env_visited[state_id]:
                    step_loss.append(loss_weights * torch.stack([loss_component[env_index] for loss_component in step.L]))
                else:
                    env_visited[state_id] = True

            if step_loss:
                step_loss = torch.mean(torch.stack(step_loss, dim=0), dim=0)
                tem_loss = tem_loss + torch.sum(step_loss)

            logits, _ = predictor.predict_next_logits(step.g_inf, step.a)
            labels = next_state_labels[step_index]
            supervised_loss = supervised_loss + supervised_criterion(logits, labels)
            accuracy_correct += (logits.argmax(dim=1) == labels).sum().item()
            accuracy_total += labels.numel()

        tem_loss = tem_loss / rollout_length
        supervised_loss = supervised_loss / rollout_length
        total_loss = tem_weight * tem_loss + supervised_weight * supervised_loss

        total_loss.backward()
        optimizer.step()

        if train_step % 10 == 0 or train_step == train_steps - 1:
            accuracy = 100.0 * accuracy_correct / max(accuracy_total, 1)
            print(
                f"step {train_step + 1:05d}/{train_steps:05d} "
                f"| total={total_loss.item():.4f} "
                f"| tem={tem_loss.item():.4f} "
                f"| sup={supervised_loss.item():.4f} "
                f"| acc={accuracy:.2f}% "
                f"| device={device}"
            )

        if checkpoint_path and ((train_step + 1) % checkpoint_every == 0 or train_step == train_steps - 1):
            torch.save(
                {
                    "model_state_dict": predictor.state_dict(),
                    "train_config": {
                        "train_steps": train_steps,
                        "batch_size": batch_size,
                        "rollout_length": rollout_length,
                        "supervised_weight": supervised_weight,
                        "tem_weight": tem_weight,
                        "seed": seed,
                    },
                },
                checkpoint_path,
            )

    return predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TEM-style model on the 4x4 wine task.")
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rollout-length", type=int, default=20)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--tem-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-path", type=str, default="tem_2d_style.pt")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        rollout_length=args.rollout_length,
        supervised_weight=args.supervised_weight,
        tem_weight=args.tem_weight,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        device=args.device,
    )

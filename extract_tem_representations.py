#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract node-level g and p representations from a trained TEM-style model."""

import argparse
import random

import numpy as np
import torch

from train_2d_tem_style import (
    NUM_STATES,
    build_persistent_tem_batch,
    load_model,
    make_env_state,
)


def collect_representations(
    checkpoint_path,
    batch_size=16,
    rollout_length=20,
    num_chunks=100,
    remap_strategy="fixed",
    remap_curriculum_steps=1000,
    walk_length_min_chunks=5,
    walk_length_max_chunks=15,
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

    predictor, train_config, _ = load_model(
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        device=device,
    )
    predictor.eval()
    tem = predictor.tem

    env_states = [
        make_env_state(
            rollout_length=rollout_length,
            n_x=tem.hyper["n_x"],
            walk_length_min_chunks=walk_length_min_chunks,
            walk_length_max_chunks=walk_length_max_chunks,
        )
        for _ in range(batch_size)
    ]
    prev_iter = None

    g_dim = sum(tem.hyper["n_g"])
    p_dim = sum(tem.hyper["n_p"])

    g_sum = torch.zeros(NUM_STATES, g_dim, device=device)
    p_sum = torch.zeros(NUM_STATES, p_dim, device=device)
    counts = torch.zeros(NUM_STATES, device=device)

    with torch.no_grad():
        for chunk_index in range(num_chunks):
            remap_probability = (
                min((chunk_index + 1) / max(remap_curriculum_steps, 1), 1.0)
                if remap_strategy == "curriculum"
                else 0.0
            )
            chunk, _ = build_persistent_tem_batch(
                env_states=env_states,
                prev_iter=prev_iter,
                rollout_length=rollout_length,
                n_x=tem.hyper["n_x"],
                device=device,
                remap_strategy=remap_strategy,
                remap_probability=remap_probability,
                walk_length_min_chunks=walk_length_min_chunks,
                walk_length_max_chunks=walk_length_max_chunks,
            )
            forward = tem(chunk, prev_iter=prev_iter)

            for step in forward:
                for env_index in range(batch_size):
                    state_id = step.g[env_index]["id"]
                    g_vec = torch.cat([module[env_index] for module in step.g_inf], dim=0)
                    p_vec = torch.cat([module[env_index] for module in step.p_inf], dim=0)

                    g_sum[state_id] += g_vec
                    p_sum[state_id] += p_vec
                    counts[state_id] += 1

            prev_iter = [forward[-1].detach()]

    safe_counts = counts.clone()
    safe_counts[safe_counts == 0] = 1

    g_mean = g_sum / safe_counts.unsqueeze(1)
    p_mean = p_sum / safe_counts.unsqueeze(1)

    return {
        "g_vectors": g_mean.detach().cpu(),
        "p_vectors": p_mean.detach().cpu(),
        "counts": counts.detach().cpu(),
        "train_config": train_config,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Extract g/p node representations from a TEM-style checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rollout-length", type=int, default=20)
    parser.add_argument("--num-chunks", type=int, default=100)
    parser.add_argument("--remap-strategy", type=str, default="fixed", choices=["fixed", "resample", "curriculum"])
    parser.add_argument("--remap-curriculum-steps", type=int, default=1000)
    parser.add_argument("--walk-length-min-chunks", type=int, default=5)
    parser.add_argument("--walk-length-max-chunks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="tem_2d_representations.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    representations = collect_representations(
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        rollout_length=args.rollout_length,
        num_chunks=args.num_chunks,
        remap_strategy=args.remap_strategy,
        remap_curriculum_steps=args.remap_curriculum_steps,
        walk_length_min_chunks=args.walk_length_min_chunks,
        walk_length_max_chunks=args.walk_length_max_chunks,
        seed=args.seed,
        device=args.device,
    )
    torch.save(representations, args.output_path)
    print(f"Saved representations to {args.output_path}")

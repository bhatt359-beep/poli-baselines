from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from poli.objective_repository import EhrlichHoloProblemFactory
except ImportError:
    print("ERROR: poli library not found. Please install it with:")
    print("  pip install poli-core")
    sys.exit(1)

from poli_baselines.solvers.dyna_ppo_mutative import DynaPPOMutativeSolver


class ProgressTracker:
    """Track best score as the solver runs."""

    def __init__(self):
        self.best_scores = []

    def __call__(self, solver):
        best_value = float(np.max(solver.get_best_performance()))
        self.best_scores.append(best_value)
        if solver.iteration > 0 and solver.iteration % 10 == 0:
            print(f"  Iteration {solver.iteration:3d}: best score = {best_value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mutative DyNA PPO on Ehrlich Holo.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--motif-length", type=int, default=8)
    parser.add_argument("--n-motifs", type=int, default=4)
    parser.add_argument("--plot-path", type=Path, default=Path("score_per_iteration_mutative.png"))
    args = parser.parse_args()

    print("Starting mutative DyNA PPO script")
    print("=" * 60)

    problem = EhrlichHoloProblemFactory().create(
        sequence_length=args.sequence_length,
        motif_length=args.motif_length,
        n_motifs=args.n_motifs,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    print("Problem factory created (Ehrlich Holo optimization)")
    print(f"  Initial population size: {len(x0)}")
    print(f"  Initial best score: {float(y0.max()):.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    solver = DynaPPOMutativeSolver(
        black_box=f,
        x0=x0,
        y0=y0,
        device=device,
        max_mutation_steps=8,
        num_experiment_rounds=1,
        num_model_rounds=2,
        actor_lr=1e-5,
        critic_lr=1e-5,
        density_penalty_weight=0.0,
        elite_fraction=0.2,
        greedy_epsilon=0.2,
    )

    print("Mutative DyNA PPO solver initialized")
    print("=" * 60)

    tracker = ProgressTracker()
    solver.solve(
        max_iter=args.iterations,
        verbose=False,
        post_step_callbacks=[tracker],
    )

    xs_history, ys_history = solver.get_history_as_arrays()
    ys_history = np.squeeze(ys_history).astype(float)
    finite_vals = ys_history[np.isfinite(ys_history)]
    if len(finite_vals) > 0:
        ys_history[~np.isfinite(ys_history)] = finite_vals.min()
    else:
        ys_history[~np.isfinite(ys_history)] = -1.0

    best_solution = solver.get_best_solution(top_k=1)
    best_score = float(np.nanmax(ys_history)) if len(ys_history) > 0 else float(y0.max())

    print("\nResults")
    print("=" * 60)
    if len(best_solution) > 0:
        print(f"Best sequence found: {best_solution[0]}")
        print(f"Best score:         {best_score:.4f}")
        print(f"Improvement:        +{best_score - float(y0.max()):.4f}")
    else:
        print("No solution found")

    if tracker.best_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(tracker.best_scores) + 1), tracker.best_scores, linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Best Score")
        plt.title("Mutative DyNA PPO on Ehrlich Holo")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {args.plot_path}")

    f.terminate()


if __name__ == "__main__":
    main()
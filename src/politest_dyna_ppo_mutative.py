from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from poli.core.util.proteins.defaults import AMINO_ACIDS

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
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--trials", type=int, default=5, help="Number of independent trials")
    parser.add_argument("--sequence-length", type=int, default=15)
    parser.add_argument("--motif-length", type=int, default=7)
    parser.add_argument("--n-motifs", type=int, default=2)
    parser.add_argument("--initial-samples", type=int, default=10, help="Number of initial samples")
    parser.add_argument('--pest-control', action='store_true')
    args = parser.parse_args()

    print("Starting mutative DyNA PPO script")
    print("=" * 60)
    print(f"Running {args.trials} trials with {args.iterations} iterations each")
    print("=" * 60)

    trial_results = []
    
    alphabet = ['A', 'C', 'T', 'G', 'U'] if args.pest_control else AMINO_ACIDS
    

    for trial_idx in range(args.trials):
        print(f"\n[Trial {trial_idx + 1}/{args.trials}]")
        
        problem = EhrlichHoloProblemFactory().create(
            sequence_length=args.sequence_length,
            motif_length=args.motif_length,
            n_motifs=args.n_motifs,
            alphabet=alphabet,
        )
        f = problem.black_box
        x0 = f.initial_solution(n_samples=args.initial_samples)
        y0 = f(x0)

        if trial_idx == 0:
            print("Problem factory created (Ehrlich Holo optimization)")
        print(f"  Initial population size: {len(x0)}")
        print(f"  Initial best score: {float(y0.max()):.4f}")
        
        problem.seed = trial_idx

        device = "mps" # "cuda" if torch.cuda.is_available() else "cpu"

        solver = DynaPPOMutativeSolver(
            black_box=f,
            x0=x0,
            y0=y0,
            device=device,
            max_mutation_steps=1,           # ← CHANGED
            greedy_epsilon=1.0,
            epsilon_decay_start=15,
            epsilon_decay_end=0.02,
            greedy=True,
            num_experiment_rounds=3,        # ← CHANGED
            num_model_rounds=0,
            ppo_epochs=2,
            batch_size=16,
            actor_lr=4.3e-4,
            critic_lr=4.3e-4,
            density_penalty_weight=0.0,
            elite_fraction=0.2,
            use_model_based=False,
            epsilon_clip=0.5,
            top_k=3                         # ← CHANGED
        )

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
        trial_results.append(best_score)

        print(f"  Trial {trial_idx + 1} best score: {best_score:.4f}")

        f.terminate()

    # Print summary statistics across all trials
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS ACROSS TRIALS")
    print("=" * 60)
    results_array = np.array(trial_results)
    print(f"\nAcross {args.trials} trials:")
    print(f"  Mean final score:   {results_array.mean():.4f}")
    print(f"  Std dev:            {results_array.std():.4f}")
    print(f"  Min score:          {results_array.min():.4f}")
    print(f"  Max score:          {results_array.max():.4f}")
    print(f"  Individual scores:  {[f'{s:.4f}' for s in trial_results]}")
    print()
    print("Done!")



if __name__ == "__main__":
    main()
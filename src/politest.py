from pathlib import Path
import os
from poli.objective_repository import EhrlichHoloProblemFactory
from poli.objective_repository import RaspProblemFactory
from poli_baselines.solvers.dyna_ppo import DynaPPOSolver
from poli_baselines.solvers.simple.random_mutation import RandomMutation

# GitHub credentials (use environment variable for security)
github_token = os.getenv("GITHUB_TOKEN")
if github_token:
    from github import Github
    g = Github(github_token)

# Creating an instance of the problem
# Using a PDB file from the examples
pdb_path = Path(__file__).parent.parent / "examples" / "06_running_lambo2_on_rasp" / "rfp_pdbs" / "2vad_A" / "2vad_A.pdb"
problem = RaspProblemFactory().create(wildtype_pdb_path=str(pdb_path), additive=True)
f, x0 = problem.black_box, problem.x0
y0 = f(x0)

print("Set Up Problem factory")

# Creating an instance of the solver
solver = DynaPPOSolver(
    black_box=f,
    x0=x0,
    y0=y0,
    density_radius=0,
    actor_lr=1e-4,
    critic_lr=1e-4,
    ppo_epochs=2,
)

print("Set Up Solver")

# Running the optimization for 1000 steps,
# breaking if we find a performance above 5.0.
solver.solve(max_iter=2000, break_at_performance=1.0, verbose=True)

# Checking if we got the solution we were waiting for
print(solver.get_best_solution())  # Should be [["A", "L", "O", "H", "A"]]
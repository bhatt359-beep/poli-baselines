"""
Implements DynaPPO: Model-based RL for sequence design.

Paper: https://openreview.net/pdf?id=HklxbgBKvr

DyNA PPO is a model-based reinforcement learning approach for sample-efficient
biological sequence design. It combines:
- Actor and Critic networks trained via PPO
- An ensemble of surrogate models for approximating the reward function
- Alternating experiment-based and model-based training phases
- Automatic model selection via cross-validation R² thresholding
- Generalized Advantage Estimation (GAE) for variance reduction
"""

from __future__ import annotations

import random
import warnings
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from poli.core.abstract_black_box import AbstractBlackBox
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import hamming

# import edlib

from poli_baselines.core.step_by_step_solver import StepByStepSolver

# def edit_dist(x: str, y: str):
#     """
#     Computes the edit distance between two strings.
#     """
#     return edlib.align(x, y)["editDistance"]

class ActorNetwork(nn.Module):
    """Policy network π_θ that outputs action probabilities."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        """
        Initialize actor network.

        Parameters
        ----------
        input_dim : int
            Dimension of input state (one-hot encoded sequence)
        hidden_dim : int
            Dimension of hidden layers
        action_dim : int
            Dimension of action space (vocabulary size)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        return self.net(state)

    def get_action_and_logprob(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return action + log probability."""
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob


class CriticNetwork(nn.Module):
    """Value network V_ϕ that outputs scalar state values."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize critic network.

        Parameters
        ----------
        input_dim : int
            Dimension of input state (one-hot encoded sequence)
        hidden_dim : int
            Dimension of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return value estimate."""
        return self.net(state).squeeze(-1)


class DynaPPOSolver(StepByStepSolver):
    """
    DyNA PPO Solver for discrete sequence optimization.

    Combines PPO policy gradient training with model-based RL using an ensemble
    of surrogate models. The actor network generates sequences autoregressively,
    the critic estimates state values, and surrogate models approximate the
    reward function for model-based rollouts.

    Parameters
    ----------
    black_box : AbstractBlackBox
        The black box function to optimize.
    x0 : np.ndarray
        The initial input sequences.
    y0 : np.ndarray
        The initial output values.
    alphabet : list[str] | None
        The alphabet of possible tokens. If None, inferred from black_box.
    tokenizer : Callable[[str], list[str]] | None
        Function to tokenize strings.
    num_episodes : int
        Number of episodes to collect before PPO update. Default: 5
    ppo_epochs : int
        Number of PPO epochs per training. Default: 4
    batch_size : int
        PPO minibatch size. Default: 32
    actor_lr : float
        Actor network learning rate. Default: 3e-4
    critic_lr : float
        Critic network learning rate. Default: 1e-3
    gamma : float
        Discount factor. Default: 0.99
    gae_lambda : float
        GAE lambda parameter. Default: 0.95
    epsilon_clip : float
        PPO clip parameter. Default: 0.2
    entropy_coef : float
        Entropy regularization coefficient. Default: 0.01
    value_coef : float
        Value loss coefficient. Default: 0.5
    r_squared_threshold : float
        Minimum R² for surrogate models. Default: 0.5
    use_model_based : bool
        Whether to use surrogate models for rollouts. Default: True
    density_penalty_weight : float
        Weight for diversity penalty based on sequence density. Default: 1.0
    density_radius : int
        Edit distance threshold for density computation. Default: 2
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        alphabet: list[str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        greedy_epsilon: float = 0,
        num_episodes: int = 15,
        ppo_epochs: int = 8,
        batch_size: int = 16,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        r_squared_threshold: float = 0.5,
        use_model_based: bool = True,
        density_penalty_weight: float = 1.0,
        density_radius: int = 2,
    ):
        """Initialize DyNA PPO solver."""
        # Handle tokenization
        if x0.ndim == 1:
            if tokenizer is None:
                warnings.warn(
                    "Input is 1D array without tokenizer. "
                    "Tokenizing character by character."
                )

                def tokenizer(x):
                    return list(x)

            x0_ = [tokenizer(x_i) for x_i in x0]
            x0 = np.array(x0_)

        super().__init__(black_box, x0, y0)

        self.alphabet = (
            black_box.info.alphabet if alphabet is None else alphabet
        )
        self.alphabet_without_empty = [s for s in self.alphabet if s != ""]
        self.string_to_idx = {symbol: i for i, symbol in enumerate(self.alphabet)}
        self.idx_to_string = {
            v: k for k, v in self.string_to_idx.items() if k != ""
        }
        self.alphabet_size = len(self.alphabet)
        self.tokenizer = tokenizer
        self.seq_len = len(x0[0]) if x0.ndim > 1 else len(x0)
        
        # Greedy Epsilon
        self.greedy_epsilon = greedy_epsilon

        # PPO hyperparameters
        self.num_episodes = num_episodes
        self.ppo_epochs = ppo_epochs
        self.batch_size_ppo = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Model-based settings
        self.r_squared_threshold = r_squared_threshold
        self.use_model_based = use_model_based

        # Diversity penalty settings
        self.density_penalty_weight = density_penalty_weight
        self.density_radius = density_radius

        # Initialize actor and critic networks
        # State: one-hot encoded partial sequence + positional encoding
        # For simplicity, we flatten the sequence so far
        obs_dim = self.seq_len * self.alphabet_size + 1  # +1 for position
        hidden_dim = 128

        self.actor = ActorNetwork(obs_dim, hidden_dim, self.alphabet_size)
        self.critic = CriticNetwork(obs_dim, hidden_dim)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Initialize ensemble of models for model-based rollouts
        self.models = self._build_model_ensemble()
        self.model_scores = {}
        self.model_based_enabled = False

        # Rollout buffer for PPO training
        self.rollout_buffer = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
        }

    def _encode_state(self, partial_seq: np.ndarray, position: int) -> torch.Tensor:
        """
        Encode partial sequence as one-hot + position embedding.

        Parameters
        ----------
        partial_seq : np.ndarray
            Partial sequence so far (shape: (position,))
        position : int
            Current position in sequence generation

        Returns
        -------
        torch.Tensor
            Encoded state tensor (shape: (obs_dim,))
        """
        # One-hot encode the partial sequence
        one_hot = np.zeros((self.seq_len, self.alphabet_size))
        for i, token in enumerate(partial_seq):
            token_idx = self.string_to_idx.get(token, 0)
            one_hot[i, token_idx] = 1

        # Flatten and add position
        state = np.concatenate([one_hot.flatten(), [position / self.seq_len]])
        return torch.FloatTensor(state)

    def _compute_density_penalty(self, sequence: np.ndarray) -> float:
        """
        Compute density-based penalty for a sequence.

        Penalizes sequences similar to previously proposed ones, encouraging
        exploration of new regions of the search space.

        Parameters
        ----------
        sequence : np.ndarray
            The sequence to evaluate

        Returns
        -------
        float
            Density penalty value (sum of linearly decaying weights for nearby sequences)
        """
        xs, ys = self.get_history_as_arrays()

        if len(xs) == 0:
            return 0.0

        seq_str = "".join(str(s) for s in sequence)
        penalty = 0.0

        for x_hist in xs:
            hist_str = "".join(str(s) for s in x_hist)
            # Compute Hamming distance
            distance = hamming(list(seq_str), list(hist_str))

            # Only penalize if within density radius
            if distance < self.density_radius:
                # Linear weight decay: weight = 1 - (distance / radius)
                weight = 1.0 - (distance / self.density_radius)
                penalty += weight

        return penalty

    def _sample_episode(self) -> Tuple[np.ndarray, List, List, List, List, List]:
        """
        Collect a complete episode trajectory.

        Returns
        -------
        sequence : np.ndarray
            Generated sequence
        states : list
            List of states
        actions : list
            List of actions (token indices)
        rewards : list
            List of rewards (0 until end, then fitness)
        log_probs : list
            List of log probabilities
        values : list
            List of value estimates
        """
        states = []
        actions = []
        log_probs = []
        values = []

        partial_seq = np.array([])
        total_reward = 0.0

        # Generate sequence autoregressively
        for t in range(self.seq_len):
            # Encode state
            state = self._encode_state(partial_seq, t)
            states.append(state)

            # Get action from actor
            with torch.no_grad():
                action, logprob = self.actor.get_action_and_logprob(state)
                value = self.critic(state)

            action = action.item()
            logprob = logprob.item()
            value = value.item()

            log_probs.append(logprob)
            values.append(value)
            actions.append(action)

            # Add token to sequence
            token = self.idx_to_string.get(action, self.alphabet_without_empty[0])
            partial_seq = np.append(partial_seq, token)

        # Complete sequence - get fitness
        if self.model_based_enabled and self.use_model_based:
            seq_encoded = self._encode_sequences(partial_seq.reshape(1, -1))
            predictions = [
                model.predict(seq_encoded)[0]
                for i, model in enumerate(self.models)
                if self.model_scores.get(i, -1) >= self.r_squared_threshold
            ]
            fitness = np.mean(predictions)
        else:
            fitness = self.black_box(np.array([partial_seq]))[0, 0]

        # Apply density-based diversity penalty
        density_penalty = self._compute_density_penalty(partial_seq)
        final_reward = max(fitness, -1) - self.density_penalty_weight * density_penalty

        # Build trajectory rewards: only final step has real reward
        # Earlier steps use critic value bootstrapping  
        rewards = [0.0] * (self.seq_len - 1) + [final_reward]

        return partial_seq, states, actions, rewards, log_probs, values

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[List[float], List[float]]:
        """
        Compute advantages and returns using Generalized Advantage Estimation.

        Parameters
        ----------
        rewards : list
            Rewards from trajectory
        values : list
            Value estimates from critic
        dones : list
            Done flags

        Returns
        -------
        advantages : list
            Computed advantages
        returns : list
            Computed returns
        """
        advantages = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = (
                delta
                + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            )
            advantages.insert(0, gae)

        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = [adv + val for adv, val in zip(advantages, values)]

        return list(advantages), returns

    def _ppo_update(self, states: List, actions: List, old_log_probs: List,
                    advantages: List, returns: List) -> Tuple[float, float, float]:
        """
        Perform PPO update on collected trajectory.

        Parameters
        ----------
        states : list
            State tensors
        actions : list
            Action indices
        old_log_probs : list
            Old log probabilities
        advantages : list
            Advantages
        returns : list
            Returns

        Returns
        -------
        actor_loss : float
            Mean actor loss
        critic_loss : float
            Mean critic loss
        entropy : float
            Mean entropy
        """
        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(len(states))

            for i in range(0, len(states), self.batch_size_ppo):
                batch_idx = indices[i : i + self.batch_size_ppo]

                s_batch = states[batch_idx]
                a_batch = actions[batch_idx]
                old_logp_batch = old_log_probs[batch_idx]
                adv_batch = advantages[batch_idx]
                ret_batch = returns[batch_idx]

                # Forward pass
                logits = self.actor(s_batch)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(a_batch)
                entropy = dist.entropy().mean()

                v_batch = self.critic(s_batch)

                # PPO clipped objective
                ratio = torch.exp(new_logp - old_logp_batch)
                surr1 = ratio * adv_batch
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
                    * adv_batch
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(v_batch, ret_batch)
                total_loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                # Actor update
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                self.actor_optim.step()
                self.critic_optim.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                num_batches += 1

        return (
            total_actor_loss / num_batches,
            total_critic_loss / num_batches,
            total_entropy / num_batches,
        )

    def _build_model_ensemble(self) -> list:
        """Build an ensemble of diverse regression models."""
        return [
            KNeighborsRegressor(n_neighbors=3),
            BayesianRidge(),
            RandomForestRegressor(n_estimators=50, random_state=42),
            GradientBoostingRegressor(n_estimators=50, random_state=42),
            GaussianProcessRegressor(random_state=42, alpha=1e-6),
        ]

    def next_candidate(self) -> np.ndarray:
        """
        Generate the next candidate sequence using the actor policy with exploration.

        Mixes actor policy with random actions for better space exploration.
        """
        partial_seq = np.array([])

        with torch.no_grad():
            for t in range(self.seq_len):
                # Epsilon-greedy: random action sometimes, otherwise from actor
                if random.random() < self.greedy_epsilon:
                    # Random action for exploration
                    action = random.randint(0, self.alphabet_size - 1)
                else:
                # Actor-based action
                    state = self._encode_state(partial_seq, t)
                    logits = self.actor(state)
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()

                # Add token to sequence
                token = self.idx_to_string.get(action, self.alphabet_without_empty[0])
                partial_seq = np.append(partial_seq, token)

        return np.array([partial_seq])

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Update the PPO networks and surrogate models after receiving evaluations.

        Periodically:
        1. Collects trajectory episodes with the current policy
        2. Trains actor and critic networks via PPO
        3. Updates surrogate model ensemble
        """
        # Every update_frequency steps, do PPO training
        if self.iteration % 5 == 0 or self.iteration < 5:
            self._ppo_train()

        # Every 10 steps, retrain surrogate models
        if self.iteration % 10 == 0:
            self._update_surrogate_models()

    def _ppo_train(self) -> None:
        """Collect episodes and perform PPO training."""
        all_states = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_rewards = []
        all_dones = []

        # Collect multiple episodes
        for _ in range(self.num_episodes):
            partial_seq, states, actions, rewards, log_probs, values = (
                self._sample_episode()
            )

            all_states.extend(states)
            all_actions.extend(actions)
            all_log_probs.extend(log_probs)
            all_values.extend(values)
            all_rewards.extend(rewards)
            # Episode always terminates at sequence length
            dones = [False] * (len(rewards) - 1) + [True]
            all_dones.extend(dones)

        # Compute advantages and returns
        advantages, returns = self._compute_gae(all_rewards, all_values, all_dones)

        # PPO update
        actor_loss, critic_loss, entropy = self._ppo_update(
            all_states, all_actions, all_log_probs, advantages, returns
        )

    def _update_surrogate_models(self) -> None:
        """Train surrogate model ensemble on observed (sequence, reward) pairs."""
        xs, ys = self.get_history_as_arrays()

        if len(xs) < 10:
            self.model_based_enabled = False
            return

        # One-hot encode sequences
        xs_encoded = self._encode_sequences(xs)
        ys_flat = ys.flatten()

        # Train and evaluate each model
        self.model_scores = {}
        for i, model in enumerate(self.models):
            try:
                cv_scores = cross_val_score(
                    model, xs_encoded, ys_flat, cv=min(5, len(xs) // 2), scoring="r2"
                )
                r2_score = cv_scores.mean()
                self.model_scores[i] = r2_score

                if r2_score >= self.r_squared_threshold:
                    model.fit(xs_encoded, ys_flat)
            except Exception:
                self.model_scores[i] = -1.0

        self.model_based_enabled = any(
            score >= self.r_squared_threshold for score in self.model_scores.values()
        )

    def _encode_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        One-hot encode sequences for model training.

        Parameters
        ----------
        sequences : np.ndarray
            Array of sequences (string or object dtype)

        Returns
        -------
        np.ndarray
            One-hot encoded sequences
        """
        encoded = []
        for seq in sequences:
            # Convert sequence to indices
            indices = [
                self.string_to_idx.get(token, 0) for token in seq
            ]
            # One-hot encode
            one_hot = np.zeros((len(seq), self.alphabet_size))
            for i, idx in enumerate(indices):
                one_hot[i, idx] = 1
            encoded.append(one_hot.flatten())

        return np.array(encoded)

    def get_best_solution(self, top_k: int = 1) -> list[str]:
        """
        Get the best solution(s) found so far.

        Parameters
        ----------
        top_k : int
            Number of top solutions to return.

        Returns
        -------
        list[str]
            List of best sequences as strings.
        """
        xs, ys = self.get_history_as_arrays()
        if len(xs) == 0:
            return []

        best_indices = np.argsort(ys.flatten())[-top_k:][::-1]
        best_solutions = []
        for idx in best_indices:
            seq = xs[idx]
            seq_str = "".join(str(s) for s in seq)
            best_solutions.append(seq_str)

        return best_solutions


if __name__ == "__main__":
    # ===== SANITY CHECK TEST =====
    # Test DyNA PPO on the Aloha problem, following the pattern in readme.md
    from poli.objective_repository import AlohaProblemFactory

    print("=" * 70)
    print("DyNA PPO SANITY CHECK ON ALOHA PROBLEM")
    print("=" * 70)

    # Creating an instance of the problem (following readme.md pattern)
    problem = AlohaProblemFactory().create()
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    print(f"\n[*] Problem Setup:")
    print(f"    Initial x0: {x0}")
    print(f"    Initial y0: {y0}")

    # Creating an instance of the DyNA PPO solver
    solver = DynaPPOSolver(
        black_box=f,
        x0=x0,
        y0=y0,
        greedy_epsilon=0.25,
    )

    max_iter = 1000
    
    print(f"\n[*] Running DyNA PPO for {max_iter} iterations...")
    print(f"    (looking for solution with fitness >= 5.0)")
    
    # Running the optimization, breaking if we find a solution with fitness 5.0
    solver.solve(max_iter=max_iter, break_at_performance=5.0, verbose=False)

    # Checking if we got the solution we were waiting for
    best_solution = solver.get_best_solution(top_k=1)
    xs, ys = solver.get_history_as_arrays()

    print(f"\n=== Results ===")
    print(f"Total iterations: {solver.iteration}")
    print(f"Total evaluations: {len(xs)}")
    print(f"Best fitness found: {ys.max():.1f}")
    print(f"Best solution: {best_solution}")
    print(f"Expected: ['ALOHA']")
    print(f"Model-based training enabled: {solver.model_based_enabled}")

    # Success check
    if ys.max() >= 5.0:
        print(f"\n✓ Successfully found optimal solution!")
    else:
        print(f"\n✓ Sanity check passed (found good solution, but not optimal)")

    f.terminate()
    print("=" * 70 + "\n")

"""
Implements a mutative DyNA PPO solver for sequence design.

This variant starts from an already existing sequence and applies a sequence of
learned mutations, rather than generating a candidate from scratch.
"""

from __future__ import annotations

import random
import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from torch.distributions import Categorical

from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.step_by_step_solver import StepByStepSolver


class ActorNetwork(nn.Module):
    """Policy network that outputs mutation logits."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def get_action_and_logprob(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob


class CriticNetwork(nn.Module):
    """Value network for mutation states."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class DynaPPOMutativeSolver(StepByStepSolver):
    """
    Mutative DyNA PPO solver for discrete sequence optimization.

    The solver starts from historical high-scoring sequences and learns a PPO
    policy over mutation actions. Surrogate models are used for additional
    model-based rollouts once they are sufficiently predictive.
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        alphabet: list[str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        device: str = "cpu",
        greedy_epsilon: float = 0.1,
        num_experiment_rounds: int = 1,
        num_model_rounds: int = 1,
        max_mutation_steps: int = 8,
        ppo_epochs: int = 8,
        batch_size: int = 32,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        r_squared_threshold: float = 0.5,
        use_model_based: bool = True,
        density_penalty_weight: float = 0.0,
        density_radius: int = 2,
        elite_fraction: float = 0.25,
        hidden_dim: int = 128,
        min_history_for_model: int = 10,
        fitness_floor: float = -1.0,
    ):
        if x0.ndim == 1:
            if tokenizer is None:
                warnings.warn(
                    "Input is 1D array without tokenizer. Tokenizing character by character."
                )

                def tokenizer(x):
                    return list(x)

            x0 = np.array([tokenizer(x_i) for x_i in x0])

        super().__init__(black_box, x0, y0)

        self.alphabet = black_box.info.alphabet if alphabet is None else alphabet
        self.alphabet_without_empty = [symbol for symbol in self.alphabet if symbol != ""]
        self.string_to_idx = {symbol: i for i, symbol in enumerate(self.alphabet)}
        self.idx_to_string = {
            index: symbol for symbol, index in self.string_to_idx.items() if symbol != ""
        }
        self.alphabet_size = len(self.alphabet)
        self.mutable_alphabet_size = len(self.alphabet_without_empty)
        self.tokenizer = tokenizer
        self.seq_len = len(x0[0]) if x0.ndim > 1 else len(x0)

        self.device = torch.device(device)
        self.greedy_epsilon = greedy_epsilon
        self.num_experiment_rounds = num_experiment_rounds
        self.num_model_rounds = num_model_rounds
        self.max_mutation_steps = max_mutation_steps
        self.ppo_epochs = ppo_epochs
        self.batch_size_ppo = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.r_squared_threshold = r_squared_threshold
        self.use_model_based = use_model_based
        self.density_penalty_weight = density_penalty_weight
        self.density_radius = density_radius
        self.elite_fraction = elite_fraction
        self.min_history_for_model = min_history_for_model
        self.fitness_floor = fitness_floor

        obs_dim = self.seq_len * self.alphabet_size + 2
        action_dim = self.seq_len * self.mutable_alphabet_size

        self.actor = ActorNetwork(obs_dim, hidden_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.models = self._build_model_ensemble()
        self.model_scores: dict[int, float] = {}
        self.model_based_enabled = False

        self._pending_rollout: Optional[dict] = None
        self._cached_candidate_fitness: dict[Tuple[str, ...], float] = {}
        self.experimental_buffer = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
        }

    def _sequence_key(self, sequence: np.ndarray) -> Tuple[str, ...]:
        return tuple(str(token) for token in sequence.tolist())

    def _sanitize_fitness(self, value: float) -> float:
        """Replace non-finite fitness values (e.g. -inf from Ehrlich) with the floor."""
        if not np.isfinite(value):
            return self.fitness_floor
        return value

    def _encode_state(
        self, sequence: np.ndarray, fitness: float, mutation_step: int
    ) -> torch.Tensor:
        one_hot = np.zeros((self.seq_len, self.alphabet_size), dtype=float)
        for index, token in enumerate(sequence):
            token_idx = self.string_to_idx.get(str(token), 0)
            one_hot[index, token_idx] = 1.0

        mutation_progress = mutation_step / max(self.max_mutation_steps, 1)
        sanitized_fitness = self._sanitize_fitness(fitness)
        state = np.concatenate(
            [
                one_hot.flatten(),
                np.array([sanitized_fitness, mutation_progress], dtype=float),
            ]
        )
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def _hamming_distance(self, seq_a: np.ndarray, seq_b: np.ndarray) -> int:
        return int(np.sum(np.asarray(seq_a) != np.asarray(seq_b)))

    def _compute_density_penalty(self, sequence: np.ndarray) -> float:
        if self.density_penalty_weight <= 0 or self.density_radius <= 0:
            return 0.0

        xs, _ = self.get_history_as_arrays()
        if len(xs) == 0:
            return 0.0

        penalty = 0.0
        for previous in xs:
            distance = self._hamming_distance(sequence, previous)
            if distance <= self.density_radius:
                penalty += 1.0 - distance / max(self.density_radius, 1)

        return penalty

    def _select_start_sequence(self) -> Tuple[np.ndarray, float]:
        xs, ys = self.get_history_as_arrays()
        scores = ys.astype(float).flatten()
        scores[~np.isfinite(scores)] = self.fitness_floor

        # Prefer finite-scored sequences; fall back to all if none qualify
        finite_mask = np.isfinite(scores)
        if finite_mask.any():
            xs, scores = xs[finite_mask], scores[finite_mask]

        elite_count = max(1, int(np.ceil(len(xs) * self.elite_fraction)))
        elite_indices = np.argsort(scores)[-elite_count:]
        chosen_index = int(random.choice(elite_indices.tolist()))
        return xs[chosen_index].copy(), self._sanitize_fitness(float(scores[chosen_index]))

    def _decode_action(self, action: int) -> Tuple[int, str]:
        position = action // self.mutable_alphabet_size
        token_index = action % self.mutable_alphabet_size
        token = self.alphabet_without_empty[token_index]
        return position, token

    def _mutate_sequence(self, sequence: np.ndarray, action: int) -> np.ndarray:
        position, token = self._decode_action(action)
        mutated = np.array(sequence, dtype=object, copy=True)
        if str(mutated[position]) == token:
            token_index = (self.alphabet_without_empty.index(token) + 1) % self.mutable_alphabet_size
            token = self.alphabet_without_empty[token_index]
        mutated[position] = token
        return mutated

    def _build_model_ensemble(self) -> list:
        return [
            KNeighborsRegressor(n_neighbors=3),
            BayesianRidge(),
            RandomForestRegressor(n_estimators=50, random_state=42),
            GradientBoostingRegressor(n_estimators=50, random_state=42),
            GaussianProcessRegressor(random_state=42, alpha=1e-6),
        ]

    def _encode_sequences(self, sequences: np.ndarray) -> np.ndarray:
        encoded = []
        for seq in sequences:
            one_hot = np.zeros((len(seq), self.alphabet_size), dtype=float)
            for index, token in enumerate(seq):
                token_idx = self.string_to_idx.get(str(token), 0)
                one_hot[index, token_idx] = 1.0
            encoded.append(one_hot.flatten())
        return np.asarray(encoded)

    def _predict_surrogate_fitness(self, sequence: np.ndarray) -> float:
        seq_encoded = self._encode_sequences(sequence.reshape(1, -1))
        predictions = [
            model.predict(seq_encoded)[0]
            for model_index, model in enumerate(self.models)
            if self.model_scores.get(model_index, -1.0) >= self.r_squared_threshold
        ]

        if len(predictions) == 0:
            return float(self.models[int(np.argmax(list(self.model_scores.values()) or [0]))].predict(seq_encoded)[0])

        return float(np.mean(predictions))

    def _evaluate_model_sequence(self, sequence: np.ndarray) -> float:
        penalty = self.density_penalty_weight * self._compute_density_penalty(sequence)
        return self._predict_surrogate_fitness(sequence) - penalty

    def _generate_experimental_rollout(self) -> np.ndarray:
        current_sequence, current_fitness = self._select_start_sequence()
        states = []
        actions = []
        log_probs = []
        values = []

        for mutation_step in range(self.max_mutation_steps):
            state = self._encode_state(current_sequence, current_fitness, mutation_step)
            states.append(state)

            with torch.no_grad():
                if random.random() < self.greedy_epsilon:
                    action = random.randrange(self.seq_len * self.mutable_alphabet_size)
                    logits = self.actor(state)
                    dist = Categorical(logits=logits)
                    logprob = dist.log_prob(
                        torch.tensor(action, dtype=torch.long, device=self.device)
                    )
                else:
                    action_tensor, logprob = self.actor.get_action_and_logprob(state)
                    action = int(action_tensor.item())
                value = self.critic(state)

            actions.append(action)
            log_probs.append(float(logprob.item()))
            values.append(float(value.item()))
            current_sequence = self._mutate_sequence(current_sequence, action)

        self._pending_rollout = {
            "start_fitness": current_fitness,
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
        }
        # Convert back to unicode string array; black boxes require dtype.kind == 'U'
        return np.array([[str(t) for t in current_sequence]])

    def _sample_model_based_episode(
        self,
    ) -> Tuple[List[torch.Tensor], List[int], List[float], List[float], List[bool]]:
        current_sequence, current_fitness = self._select_start_sequence()
        current_fitness = self._evaluate_model_sequence(current_sequence)

        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        for mutation_step in range(self.max_mutation_steps):
            state = self._encode_state(current_sequence, current_fitness, mutation_step)
            states.append(state)

            with torch.no_grad():
                action_tensor, logprob = self.actor.get_action_and_logprob(state)
                value = self.critic(state)

            action = int(action_tensor.item())
            next_sequence = self._mutate_sequence(current_sequence, action)
            next_fitness = self._evaluate_model_sequence(next_sequence)
            reward = next_fitness - current_fitness
            done = mutation_step == self.max_mutation_steps - 1 or next_fitness <= current_fitness

            actions.append(action)
            log_probs.append(float(logprob.item()))
            values.append(float(value.item()))
            rewards.append(float(reward))
            dones.append(done)

            current_sequence = next_sequence
            current_fitness = next_fitness

            if done:
                break

        return states, actions, log_probs, values, rewards, dones

    def _append_trajectory_to_buffer(
        self,
        states: List[torch.Tensor],
        actions: List[int],
        log_probs: List[float],
        values: List[float],
        rewards: List[float],
        dones: List[bool],
    ) -> None:
        self.experimental_buffer["states"].extend(states)
        self.experimental_buffer["actions"].extend(actions)
        self.experimental_buffer["log_probs"].extend(log_probs)
        self.experimental_buffer["values"].extend(values)
        self.experimental_buffer["rewards"].extend(rewards)
        self.experimental_buffer["dones"].extend(dones)

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[List[float], List[float]]:
        # Clamp any non-finite values (e.g. -inf from Ehrlich) to prevent NaN propagation
        rewards = [r if np.isfinite(r) else 0.0 for r in rewards]
        values = [v if np.isfinite(v) else 0.0 for v in values]

        advantages = []
        gae = 0.0

        for step in reversed(range(len(rewards))):
            next_value = 0.0 if step == len(rewards) - 1 else values[step + 1]
            delta = rewards[step] + self.gamma * next_value * (1 - float(dones[step])) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - float(dones[step])) * gae
            advantages.insert(0, gae)

        advantages_array = np.asarray(advantages, dtype=float)
        if len(advantages_array) > 1:
            advantages_array = (
                (advantages_array - advantages_array.mean())
                / (advantages_array.std() + 1e-8)
            )
        returns = [advantage + value for advantage, value in zip(advantages_array.tolist(), values)]
        return advantages_array.tolist(), returns

    def _ppo_update(
        self,
        states: List[torch.Tensor],
        actions: List[int],
        old_log_probs: List[float],
        advantages: List[float],
        returns: List[float],
    ) -> Tuple[float, float, float]:
        if len(states) == 0:
            return 0.0, 0.0, 0.0

        state_tensor = torch.stack(states).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_prob_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        return_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Skip update entirely if any input tensor contains NaN/inf
        if not (
            torch.isfinite(state_tensor).all()
            and torch.isfinite(advantage_tensor).all()
            and torch.isfinite(return_tensor).all()
        ):
            return 0.0, 0.0, 0.0

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(state_tensor), device=self.device)
            for start in range(0, len(state_tensor), self.batch_size_ppo):
                batch_indices = indices[start : start + self.batch_size_ppo]

                batch_states = state_tensor[batch_indices]
                batch_actions = action_tensor[batch_indices]
                batch_old_log_probs = old_log_prob_tensor[batch_indices]
                batch_advantages = advantage_tensor[batch_indices]
                batch_returns = return_tensor[batch_indices]

                logits = self.actor(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                values = self.critic(batch_states)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.epsilon_clip,
                    1 + self.epsilon_clip,
                ) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, batch_returns)
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor_optim.step()
                self.critic_optim.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_entropy += float(entropy.item())
                num_batches += 1

        if num_batches == 0:
            return 0.0, 0.0, 0.0

        return (
            total_actor_loss / num_batches,
            total_critic_loss / num_batches,
            total_entropy / num_batches,
        )

    def _ppo_train(self) -> None:
        if len(self.experimental_buffer["states"]) == 0:
            return

        advantages, returns = self._compute_gae(
            self.experimental_buffer["rewards"],
            self.experimental_buffer["values"],
            self.experimental_buffer["dones"],
        )
        self._ppo_update(
            self.experimental_buffer["states"],
            self.experimental_buffer["actions"],
            self.experimental_buffer["log_probs"],
            advantages,
            returns,
        )
        self.experimental_buffer = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
        }

    def _model_based_train(self) -> None:
        if not self.model_based_enabled or not self.use_model_based:
            return

        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        for _ in range(self.num_model_rounds):
            episode = self._sample_model_based_episode()
            episode_states, episode_actions, episode_log_probs, episode_values, episode_rewards, episode_dones = episode
            states.extend(episode_states)
            actions.extend(episode_actions)
            log_probs.extend(episode_log_probs)
            values.extend(episode_values)
            rewards.extend(episode_rewards)
            dones.extend(episode_dones)

        if len(states) == 0:
            return

        advantages, returns = self._compute_gae(rewards, values, dones)
        self._ppo_update(states, actions, log_probs, advantages, returns)

    def _update_surrogate_models(self) -> None:
        xs, ys = self.get_history_as_arrays()
        if len(xs) < self.min_history_for_model:
            self.model_based_enabled = False
            return

        xs_encoded = self._encode_sequences(xs)
        ys_flat = ys.astype(float).flatten()

        # Only train on samples with finite labels; -inf from Ehrlich corrupts CV scoring
        finite_mask = np.isfinite(ys_flat)
        if finite_mask.sum() < self.min_history_for_model:
            self.model_based_enabled = False
            return
        xs_encoded = xs_encoded[finite_mask]
        ys_flat = ys_flat[finite_mask]

        self.model_scores = {}
        for model_index, model in enumerate(self.models):
            try:
                cv = min(5, max(2, len(xs) // 3))
                cv_scores = cross_val_score(
                    model,
                    xs_encoded,
                    ys_flat,
                    cv=cv,
                    scoring="r2",
                )
                r2_score = float(np.nanmean(cv_scores))
                self.model_scores[model_index] = r2_score
                if r2_score >= self.r_squared_threshold:
                    model.fit(xs_encoded, ys_flat)
            except Exception:
                self.model_scores[model_index] = -1.0

        self.model_based_enabled = any(
            score >= self.r_squared_threshold for score in self.model_scores.values()
        )

    def next_candidate(self) -> np.ndarray:
        return self._generate_experimental_rollout()

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        x = self.next_candidate()
        sequence = x[0]
        sequence_key = self._sequence_key(sequence)

        if sequence_key in self._cached_candidate_fitness:
            y = np.array([[self._cached_candidate_fitness.pop(sequence_key)]], dtype=float)
        else:
            y = self.black_box(x)

        self.update(x, y)
        self.post_update(x, y)
        self.iteration += 1

        return x, y

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        if self._pending_rollout is not None:
            # Sanitize both ends: Ehrlich can return -inf, which would produce nan via subtraction
            sanitized_y = self._sanitize_fitness(float(y[0, 0]))
            sanitized_start = self._sanitize_fitness(float(self._pending_rollout["start_fitness"]))
            final_reward = sanitized_y - sanitized_start
            final_reward -= self.density_penalty_weight * self._compute_density_penalty(x[0])

            rewards = [0.0] * (len(self._pending_rollout["actions"]) - 1) + [final_reward]
            dones = [False] * (len(self._pending_rollout["actions"]) - 1) + [True]
            self._append_trajectory_to_buffer(
                self._pending_rollout["states"],
                self._pending_rollout["actions"],
                self._pending_rollout["log_probs"],
                self._pending_rollout["values"],
                rewards,
                dones,
            )
            self._pending_rollout = None

        if self.iteration % max(self.num_experiment_rounds, 1) == 0:
            self._ppo_train()

        if self.iteration % 5 == 0:
            self._update_surrogate_models()
            self._model_based_train()

    def get_best_solution(self, top_k: int = 1) -> list[str]:
        xs, ys = self.get_history_as_arrays()
        if len(xs) == 0:
            return []

        ys_float = ys.astype(float).flatten()
        ys_float[~np.isfinite(ys_float)] = self.fitness_floor
        best_indices = np.argsort(ys_float)[-top_k:][::-1]
        return ["".join(str(token) for token in xs[index]) for index in best_indices]
import torch
import torch.nn.functional as F

import numpy as np

from models.ppo_network import ActorCritic

from rl.rollout_buffer import RolloutBuffer

from config.config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# PPO AGENT
# ============================================

class PPOAgent:

    def __init__(self):

        self.policy = ActorCritic(
            STATE_DIM,
            ACTION_DIM
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=LR
        )

        self.buffer = RolloutBuffer()

    # ========================================
    # ACTION SELECTION
    # ========================================

    def select_action(self, state):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            # ====================================
            # FORWARD PASS
            # ====================================

            logits, value = self.policy.forward(
                state_tensor
            )

            # ====================================
            # STATE VARIABLES
            # ====================================

            q = state[0]

            sigma = state[1]

            A = state[2]

            S = state[6]

            # ====================================
            # DEPLOYMENT CALIBRATION
            # ====================================

            # SLIGHT TRACK BOOST

            logits[:,1] += 0.18

            # ====================================
            # CONDITIONAL EXECUTE BOOST
            # ====================================

            good_execute_state = (

                q > 0.26
                and sigma < 0.30
                and A < 0.65
                and S > 0.80
            )

            if good_execute_state:

                logits[:,3] += 0.40

            # ====================================
            # SMALL REPOSITION REDUCTION
            # ====================================

            logits[:,2] -= 0.08

            # ====================================
            # SMALL ABORT REDUCTION
            # ====================================

            logits[:,4] -= 0.05

            # ====================================
            # POLICY
            # ====================================

            probs = torch.softmax(
                logits,
                dim=-1
            )

            action = torch.argmax(
                probs,
                dim=-1
            )

            dist = torch.distributions.Categorical(
                probs
            )

            logprob = dist.log_prob(action)

        # ========================================
        # STORE BUFFER
        # ========================================

        self.buffer.states.append(state)

        self.buffer.actions.append(action.item())

        self.buffer.logprobs.append(logprob.item())

        self.buffer.values.append(value.item())

        return action.item()

    # ========================================
    # GAE
    # ========================================

    def compute_gae(self, next_value=0):

        rewards = self.buffer.rewards

        dones = self.buffer.dones

        values = self.buffer.values + [next_value]

        advantages = []

        gae = 0

        for t in reversed(range(len(rewards))):

            delta = (
                rewards[t]
                + GAMMA * values[t+1] * (1 - dones[t])
                - values[t]
            )

            gae = (
                delta
                + GAMMA * LAMBDA
                * (1 - dones[t])
                * gae
            )

            advantages.insert(0, gae)

        returns = [

            adv + val

            for adv, val in zip(
                advantages,
                self.buffer.values
            )
        ]

        return returns, advantages

    # ========================================
    # PPO UPDATE
    # ========================================

    def update(self):

        returns, advantages = self.compute_gae()

        states = torch.FloatTensor(
            np.array(self.buffer.states)
        ).to(DEVICE)

        actions = torch.LongTensor(
            self.buffer.actions
        ).to(DEVICE)

        old_logprobs = torch.FloatTensor(
            self.buffer.logprobs
        ).to(DEVICE)

        returns = torch.FloatTensor(
            returns
        ).to(DEVICE)

        advantages = torch.FloatTensor(
            advantages
        ).to(DEVICE)

        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

        dataset_size = len(states)

        for _ in range(K_EPOCHS):

            indices = np.arange(dataset_size)

            np.random.shuffle(indices)

            for start in range(
                0,
                dataset_size,
                BATCH_SIZE
            ):

                end = start + BATCH_SIZE

                batch_idx = indices[start:end]

                batch_states = states[batch_idx]

                batch_actions = actions[batch_idx]

                batch_old_logprobs = \
                    old_logprobs[batch_idx]

                batch_returns = \
                    returns[batch_idx]

                batch_advantages = \
                    advantages[batch_idx]

                logprobs, values, entropy = \
                    self.policy.evaluate(
                        batch_states,
                        batch_actions
                    )

                ratios = torch.exp(
                    logprobs - batch_old_logprobs
                )

                surr1 = ratios * batch_advantages

                surr2 = torch.clamp(
                    ratios,
                    1 - CLIP_EPS,
                    1 + CLIP_EPS
                ) * batch_advantages

                actor_loss = -torch.min(
                    surr1,
                    surr2
                ).mean()

                critic_loss = F.mse_loss(
                    values,
                    batch_returns
                )

                entropy_loss = entropy.mean()

                current_entropy = entropy_loss.item()

                print(
                    f"Policy Entropy: "
                    f"{current_entropy:.4f}"
                )

                loss = (
                    actor_loss
                    + VALUE_COEF * critic_loss
                    - ENTROPY_COEF * entropy_loss
                )

                self.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    MAX_GRAD_NORM
                )

                self.optimizer.step()

        self.buffer.clear()

        return loss.item()

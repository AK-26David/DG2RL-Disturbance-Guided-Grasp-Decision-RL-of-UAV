import torch
import torch.nn as nn

class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.shared = nn.Sequential(

            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.actor = nn.Sequential(
            nn.Linear(128, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, state):

        x = self.shared(state)

        logits = self.actor(x)
        value = self.critic(x)

        return logits, value

    def act(self, state):

        logits, value = self.forward(state)

        probs = torch.softmax(logits, dim=-1)

        action = torch.argmax(
            probs,
            dim=-1
        )

        dist = torch.distributions.Categorical(probs)

        logprob = dist.log_prob(action)

        return (
            action.detach(),
            logprob.detach(),
            value.detach()
        )

    def evaluate(self, states, actions):

        logits, values = self.forward(states)

        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)

        action_logprobs = dist.log_prob(actions)

        entropy = dist.entropy()

        return (
            action_logprobs,
            values.squeeze(-1),
            entropy
        )
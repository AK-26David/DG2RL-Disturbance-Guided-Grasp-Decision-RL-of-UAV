# ============================================
# PPO MEMORY
# ============================================

class RolloutBuffer:

    def __init__(self):

        self.states = []
        self.actions = []

        self.logprobs = []

        self.rewards = []
        self.dones = []

        self.values = []

    def clear(self):

        self.states.clear()
        self.actions.clear()

        self.logprobs.clear()

        self.rewards.clear()
        self.dones.clear()

        self.values.clear()
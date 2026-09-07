import random


class BanditEnvironment:
    def __init__(self, bandit_arms):
        """
        bandit_arms: list of dictionaries with keys:
            - 'reward_outcomes'
            - 'reward_probabilities'
        """
        self.bandit_arms = bandit_arms

        self.num_actions = len(bandit_arms)

        self.state = [1.0]

        
    # Execute an action (pull a bandit arm).
    def step(self, action):
        arm = self.bandit_arms[action]
        reward = random.choices(
            arm["reward_outcomes"],
            arm["reward_probabilities"],
        )[0]

        # No transition

        return reward

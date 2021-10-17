import gym
import numpy as np


class BlackjackDefaultAgent:
    """
    This is the base class implement logic
    of OpenAI blackjack enviroment.

    The strategy learning is based on off-policy TD learning (aka Q-learning).
    """
    def __init__(self, epsilon=0.2, alpha=1e-4, gamma=1.00, env=None):
        self._env = env if env is not None else gym.make('Blackjack-v0', natural=True)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.player_hand_space = self._env.observation_space.spaces[0].n
        self.dealer_hand_space = self._env.observation_space.spaces[1].n
        self.usable_aces_space = self._env.observation_space.spaces[2].n

        self.ns = self.player_hand_space * self.dealer_hand_space * self.usable_aces_space # n possible states
        self.na = self._env.action_space.n  # n possible actions
        self.Q = np.zeros((self.ns, self.na)) # Q(s, a)

    def _get_obs_ix(self, obs):
        player, dealer, ace = obs
        return (player - 1) * self.dealer_hand_space * self.usable_aces_space + (dealer - 1) * self.usable_aces_space + ace

    def _get_greedy_policy(self):
        return self.Q.argmax(1)

    def _get_action(self, pi, s):
        return pi[s] if np.random.rand() > self.epsilon else np.random.randint(self.na)
    
    def _learn_episode(self):
        obs = self._env.reset()
        s = self._get_obs_ix(obs)
        done = False
        while not done:
            pi = self._get_greedy_policy()
            a = self._get_action(pi, s)

            obs, reward, done, _ = self._env.step(a)
            s_next = self._get_obs_ix(obs)

            self.Q[s, a] += self.alpha * (reward + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a])
            s = s_next

    def play_episode(self, pi):
        obs = self._env.reset()
        done = False
        while not done:
            s = self._get_obs_ix(obs)
            a = pi[s]
            obs, reward, done, _ = self._env.step(a)
        return reward

    def learn(self, n_train_episodes=100000, n_test_episodes=50000, verbose=10000, show=False):
        rewards = []
        for i in range(n_train_episodes):
            self._learn_episode()

            if verbose is not None and (i + 1) % verbose == 0:
                reward = self.play(n_episodes=n_test_episodes)
                rewards.append(reward)
                if show:
                    print("Score {} was achieved after {} training experiments. ".format(reward, i + 1))
        return rewards

    def play(self, n_episodes, pi=None):
        if pi is None:
            pi = self._get_greedy_policy()
        reward = 0
        for _ in range(n_episodes):
            reward += self.play_episode(pi)
        return reward / n_episodes


class BlackjackCountingAgent(BlackjackDefaultAgent):
    """
    Extends base agent model to apply it
    to the modified blackjack enviroment.
    """
    def __init__(self, epsilon=0.2, alpha=1e-4, gamma=1.00, env=None):
        super().__init__(epsilon, alpha, gamma, env)
        self.balance_space = 21  # balance is a integer value from -10 up to +10
        self.ns *= self.balance_space
        self.Q = np.zeros((self.ns, self.na))

    def _get_obs_ix(self, obs):
        player, dealer, ace, balance = obs
        return (player - 1) * self.dealer_hand_space * self.usable_aces_space * self.balance_space +\
            (dealer - 1) * self.usable_aces_space * self.balance_space + ace * self.balance_space + balance
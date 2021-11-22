from collections import namedtuple

import numpy as np

from tictactoe import TicTacToe


class QAgent:
    def __init__(self, epsilon=0.5, alpha=0.1, gamma=1.0, env=None, seed=42):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        np.random.seed(seed)

        self.env = env if env is not None else TicTacToe(3, 3, 3)
        self.Q = {}

    def get_action_greedy(self, state):
        self.check_Q(state)
        return np.argmax(self.Q[state])

    def get_action(self, state, na):
        if np.random.random() < self.epsilon:
            return np.random.randint(na)
        else:
            return self.get_action_greedy(state)

    @staticmethod
    def get_n_action_from_state(state):
        return state.count('1')

    def check_Q(self, state):
        if state not in self.Q:
            n_actions = self.get_n_action_from_state(state)
            self.Q[state] = np.zeros(n_actions)

    # self.Q[s, a] += self.alpha * (reward + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a])
    def update(self, state, action, reward, next_state=None):
        if next_state is None:
            next_Q = 0
        else:
            self.check_Q(next_state)
            next_Q = np.max(self.Q[next_state])
        self.check_Q(state)
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * next_Q - self.Q[state][action])

    def _learn_episode(self):
        self.env.reset()
        done = False
        # perform first moves for crosses and naughts
        state_prev, actions_prev = self.env.getHash(), self.env.getEmptySpaces()
        action_prev = self.get_action(state_prev, len(actions_prev))
        (state_curr, actions_curr, turn), reward, done, _ = self.env.step(actions_prev[action_prev])
        
        while not done:
            action_curr = self.get_action(state_curr, len(actions_curr))
            (state_next, actions_next, turn), reward, done, _ = self.env.step(actions_curr[action_curr])
            if done:
                break
            self.update(state_prev, action_prev, reward, state_next)
            state_prev, actions_prev = state_curr, actions_curr
            state_curr, actions_curr = state_next, actions_next
            action_prev = action_curr

        # update last crosses and naughts Q function states based on result
        self.update(state_prev, action_prev, reward * turn, None)
        self.update(state_curr, action_curr, -reward * turn, None)


    def learn(self, n_train_episodes=100000, n_test_episodes=1000, verbose=10000, min_eps=0.1):
        Res = namedtuple('result', ['iter', 'x', 'o'])
        rewards_x = []
        rewards_o = []
        n_iter = []

        eps_decay = (self.epsilon - min_eps) / n_train_episodes

        for i in range(n_train_episodes):
            self._learn_episode()
            self.epsilon -= eps_decay

            if verbose is not None and (i + 1) % verbose == 0:
                test_reward = self.play(n_episodes=n_test_episodes, random_x=False)
                rewards_x.append(test_reward)
                test_reward = self.play(n_episodes=n_test_episodes, random_o=False)
                rewards_o.append(test_reward)
                n_iter.append(i)
        print("Final X win rate: ",  rewards_x[-1][1])
        print("Final O win rate: ",  rewards_o[-1][-1])
        return Res(n_iter, rewards_x, rewards_o)


    def play_episode(self, random_x=True, random_o=True):
        self.env.reset()
        done = False
        state, actions, turn = self.env.getHash(), self.env.getEmptySpaces(), self.env.curTurn
        while not done:
            action = None
            if turn == 1:
                if random_x:
                    action = np.random.randint(len(actions))
                else:
                    action = self.get_action_greedy(state)
            elif turn == -1:
                if random_o:
                    action = np.random.randint(len(actions))
                else:
                    action = self.get_action_greedy(state)
            else:
                raise ValueError("There are only 2 players in the game, turn has to either +1 or -1")
            (state, actions, turn), reward, done, _ = self.env.step(actions[action])
        return reward

    def play(self, n_episodes=1000, random_x=True, random_o=True):
        reward = {0: 0, -1: 0, 1: 0, -10: 0}
        for _ in range(n_episodes):
            reward[self.play_episode(random_x, random_o)] += 1
        for k, _ in reward.items():
            reward[k] /= n_episodes
        return reward

from copy import deepcopy
from collections import namedtuple

import numpy as np

from tictactoe import TicTacToe


# https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
class MCTSTreeNode:
    
    def __init__(self, env, board, parent=None, constant=1):
        self.env = env
        self.n_rows = self.env.n_rows
        self.n_cols = self.env.n_cols
        self.n_win = self.env.n_win

        self.env = TicTacToe(env.n_rows, env.n_cols, env.n_win)
        self.env.board = deepcopy(board)
        self.env.isTerminal()

        self.parent = parent
        self.children = []
        self.n = 0
        self.q = 0
        self.constant = constant
        self.actions = list(self.env.getEmptySpaces())

    def select_child(self, greedy=True):
        # 1. selection
        mask = 1 - int(greedy)  # skip normalizing if pick currently best strategy
        weights = [
            (-child.q / child.n) + mask * self.constant * np.sqrt((2 * np.log(self.n) / child.n))
            for child in self.children
        ]
        return np.argmax(weights)

    def expand(self):
        # 2. expansion
        action = self.actions.pop(0)
        board = deepcopy(self.env.board)
        board[action[0], action[1]] = self.env.curTurn
        child_node = MCTSTreeNode(self.env, board, parent=self)
        child_node.env.curTurn = -self.env.curTurn
        self.children.append(child_node)
        return child_node

    def rollout(self):
        # 3. random rollout ~= simulation
        child_env = TicTacToe(self.env.n_rows, 
                                self.env.n_cols, 
                                self.env.n_win)
        child_env.board = deepcopy(self.env.board)
        child_env.curTurn = self.env.curTurn
        reward = child_env.isTerminal()
        done = child_env.gameOver
        random_actions = list(np.random.permutation(child_env.getEmptySpaces()))
        while not done:
            action = random_actions.pop(0)
            _, reward, done, _ = child_env.step(action)
        return reward * self.env.curTurn

    def backpropagate(self, result):
        # 4. backpropagation
        self.n += 1
        self.q += result
        if self.parent:
            self.parent.backpropagate(-result)


class MCTSAgent:
    def __init__(self, env=None):
        self.env = env if env is not None else TicTacToe(3, 3, 3)

    def __state_to_array(self, state):
        return np.array([int(x) - 1 for x in state]).reshape(self.env.n_rows, self.env.n_cols)
        
    def get_action_greedy(self, state, n_simulations=100):
        board = self.__state_to_array(state)
        
        tree_root = MCTSTreeNode(self.env, board)
        for _ in range(n_simulations):            
            vertex = tree_root
            while not vertex.env.gameOver:
                if len(vertex.actions) > 0:
                    vertex = vertex.expand()
                else:
                    vertex = vertex.children[vertex.select_child()]
            reward = vertex.rollout()
            vertex.backpropagate(reward)
        return tree_root.select_child(greedy=True)


    def play_episode(self, random_x=True, random_o=True):
        self.env.reset()
        done = False
        state, actions_, turn = self.env.getHash(), self.env.getEmptySpaces(), self.env.curTurn
        while not done:
            action = None
            if turn == 1:
                if random_x:
                    action = np.random.randint(len(actions_))
                else:
                    action = self.get_action_greedy(state)
            elif turn == -1:
                if random_o:
                    action = np.random.randint(len(actions_))
                else:
                    action = self.get_action_greedy(state)
            else:
                raise ValueError("There are only 2 players in the game, turn has to either +1 or -1")
            (state, actions_, turn), reward, done, _ = self.env.step(actions_[action])
        return reward


    def play(self, n_episodes=10, random_x=True, random_o=True):
        reward = {0: 0, -1: 0, 1: 0, -10: 0}
        for _ in range(n_episodes):
            reward[self.play_episode(random_x, random_o)] += 1
        for k, _ in reward.items():
            reward[k] /= n_episodes
        return reward

    def learn(self, n_test_episodes=10):
        """
        Learn nothing. Just implement to have similar methods as before.
        """
        Res = namedtuple('result', ['iter', 'x', 'o'])
        rewards_x = []
        rewards_o = []
        n_iter = []

        test_reward = self.play(n_episodes=n_test_episodes, random_x=False)
        rewards_x.append(test_reward)
        test_reward = self.play(n_episodes=n_test_episodes, random_o=False)
        rewards_o.append(test_reward)
        n_iter.append(0)
        print("Final X win rate: ",  rewards_x[-1][1])
        print("Final O win rate: ",  rewards_o[-1][-1])
        return Res(n_iter, rewards_x, rewards_o)
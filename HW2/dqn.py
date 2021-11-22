from collections import namedtuple

import random
import numpy as np
import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from tictactoe import TicTacToe


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, exptuple):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exptuple
        self.position = (self.position + 1) % self.capacity
       
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class AgentDQN:
    def __init__(
        self,
        model,
        env=None,
        epsilon=0.9,
        gamma=0.8,
        buffer_size=1000000,
        batch_size=512,
        learning_rate=1e-4,
        device='cpu'):

        self.env = env if env is not None else TicTacToe(3, 3, 3)
        self.device = device

        # -1: noughts, +1 crosses
        self.models = {-1: model().to(device), 1: model().to(device)}
        self.memory = {-1: ReplayMemory(buffer_size), 1: ReplayMemory(buffer_size)}
        self.optimizers = {-1: optim.Adam(self.models[-1].parameters(), lr=learning_rate, weight_decay=0.001),
                            1: optim.Adam(self.models[1].parameters(), lr=learning_rate, weight_decay=0.001)} 
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size


    def __state_to_tensor(self, state):
        size = self.env.n_cols
        state = np.array([int(x) for x in state])
        sx =  np.where(state > 1, 1, 0).reshape(size, size)
        so =  np.where(state < 1, 1, 0).reshape(size, size)
        se = np.where(state == 1, 1, 0).reshape(size, size)
        return torch.Tensor(np.stack([sx, so, se])).reshape(3, size, size)

    def get_action_greedy(self, state, cur_turn):
        return self.models[cur_turn](state.unsqueeze(0)).data.max(1)[1].view(1, 1)

    def get_action(self, state, cur_turn):
        if np.random.random() > self.epsilon:
            return self.get_action_greedy(state, cur_turn)
        else:
            return torch.tensor([[random.randrange(self.env.n_rows * self.env.n_rows)]], dtype=torch.int64)
        
    def __run_episode(self, do_learning=True):
        self.env.reset()
        done = False

        self.prev_states = {-1: None, 1: None}
        self.prev_actions = {}
        state, cur_turn = self.env.getHash(), self.env.curTurn

        while not done:
            state_tensor = self.__state_to_tensor(state)
            with torch.no_grad():
                action_idx = self.get_action(state_tensor.to(self.device), cur_turn).cpu()

            # Сохраняем позицию и совершаемое действие
            self.prev_states[cur_turn] = state_tensor
            self.prev_actions[cur_turn] = action_idx
            action = self.env.action_from_int(action_idx.numpy()[0][0])
            (next_state, _, cur_turn), reward, done, _ = self.env.step(action)
            next_state_tensor = self.__state_to_tensor(next_state)

            if reward == -10:
                transition = (state_tensor, action_idx,
                next_state_tensor, torch.tensor([reward], dtype=torch.float32))
                self.memory[cur_turn].store(transition)
            else:
                if self.prev_states[cur_turn] is not None:
                    if reward == -cur_turn:
                        transition = (self.prev_states[-cur_turn], 
                                      self.prev_actions[-cur_turn], 
                                      next_state_tensor, 
                                      torch.tensor([1.0], dtype=torch.float32)
                                     )
                        self.memory[-cur_turn].store(transition)
                    
                    transition = (self.prev_states[cur_turn], 
                                    self.prev_actions[cur_turn], 
                                    next_state_tensor, 
                                    torch.tensor([reward * cur_turn], dtype=torch.float32)
                                    )
                    self.memory[cur_turn].store(transition)
            
            if do_learning:
                self.update(cur_turn)
            state = next_state


    def update(self, cur_turn):
        # .learn method for 05-policygradient.ipynb
        if len(self.memory[cur_turn]) < self.batch_size:
            return
        
        # берём мини-батч из памяти
        transitions = self.memory[cur_turn].sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = Variable(torch.stack(batch_state).to(self.device))
        batch_action = Variable(torch.cat(batch_action).to(self.device))
        batch_reward = Variable(torch.cat(batch_reward).to(self.device))
        batch_next_state = Variable(torch.stack(batch_next_state).to(self.device))
        
        # считаем значения функции Q
        Q = self.models[cur_turn](batch_state)
        Q = Q.gather(1, batch_action).reshape([self.batch_size])
        
        # оцениваем ожидаемые значения после этого действия
        Qmax = self.models[cur_turn](batch_next_state).detach()
        Qmax = Qmax.max(1)[0]
        Qnext = batch_reward + (self.gamma * Qmax)
        # и хотим, чтобы Q было похоже на Qnext -- это и есть суть Q-обучения
        loss = F.smooth_l1_loss(Q, Qnext)
        self.optimizers[cur_turn].zero_grad()
        loss.backward()
        self.optimizers[cur_turn].step()

    
    def _learn_episode(self):
        self.__run_episode(do_learning=True)


    def play_episode(self, random_x=True, random_o=True):
        self.env.reset()
        state, empty_spaces, turn = self.env.getState()
        done = False
        while not done:
            action = None
            if turn == 1:
                if random_x:
                    idx = np.random.randint(len(empty_spaces))
                    action = empty_spaces[idx] 
                else:
                    idx = self.get_action_greedy(self.__state_to_tensor(state).to(self.device), turn)
                    action = self.env.action_from_int(idx)
            elif turn == -1:
                if random_o:
                    idx = np.random.randint(len(empty_spaces))
                    action = empty_spaces[idx] 
                else:
                    idx = self.get_action_greedy(self.__state_to_tensor(state).to(self.device), turn)
                    action = self.env.action_from_int(idx)
            (state, empty_spaces, turn), reward, done, _ = self.env.step(action)
        return reward


    def play(self, n_episodes=1000, random_x=True, random_o=True):
        reward = {0: 0, -1: 0, 1: 0, -10: 0}
        for _ in range(n_episodes):
            reward[self.play_episode(random_x, random_o)] += 1
        for k, _ in reward.items():
            reward[k] /= n_episodes
        return reward


    def learn(self, n_train_episodes=1000, n_test_episodes=1000, verbose=10000, min_eps=0.1):
        Res = namedtuple('result', ['iter', 'x', 'o'])
        rewards_x = []
        rewards_o = []
        n_iter = []

        eps_decay = (self.epsilon - min_eps) / n_train_episodes

        for i in tqdm.tqdm(range(n_train_episodes)):
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
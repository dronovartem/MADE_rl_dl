import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    return float(a > b) - float(a < b)

def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackJackWithCounting(gym.Env):
    def __init__(self, natural=True):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.seed()
        
        self.natural = natural
        
        self.deck = []
        self.balance = 0
        
        # https://ru.wikipedia.org/wiki/%D0%91%D0%BB%D1%8D%D0%BA%D0%B4%D0%B6%D0%B5%D0%BA
        # plus-minus strategy
        # !!! USING IT REQUIRES self.balance as a last arg of self._get_obs
        #self.weights = {
        #    2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
        #    7: 0, 8: 0, 9: 0,
        #    1: -1, 10: -1,
        #}
        # half-parts strategy
        # !!! USING IT REQUIRES int(2 * self.balance) as a last arg of self._get_obs
        self.weights = {
            2: 0.5, 3: 1, 4: 1, 5: 1.5,
            6: 1, 7: 0.5, 8: 0, 9: -0.5,
            1: -1, 10: -1,
        }
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def get_new_deck(self, n_decks=4):
        # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * n_decks

    def draw_card(self, np_random):
        index = np_random.choice(range(len(self.deck)))
        return int(self.deck.pop(index))

    def draw_hand(self, np_random):
        return [self.draw_card(np_random), self.draw_card(np_random)]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 2: #double: 1 hit & stick if possible with x2 reward
            # pick one card
            _, reward, done, _ = self.step(1)
            # if not done, i can just stick
            if not done:
                _, reward, done, _ = self.step(0)
            reward *= 2
        elif action == 1:  # hit: add a card to players hand and return
            card = self.draw_card(self.np_random)
            self.balance += self.weights[card] 
            self.player.append(card)
            if is_bust(self.player):
                done = True
                reward = -1.
                self.balance += self.weights[self.dealer[1]] 
            else:
                done = False
                reward = 0.
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                card = self.draw_card(self.np_random)
                self.dealer.append(card)
                self.balance += self.weights[card]
            # dealer took what he need, now he can show me all cards (hidden one too)
            self.balance += self.weights[self.dealer[1]]

            # define winner etc
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
                
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # `int(2 * self.balance)` if parts or `self.balance` if plus-minus memory
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), int(2 * self.balance))

    def reset(self, min_count=15):
        if len(self.deck) < min_count:
            self.get_new_deck()
            self.balance = 0

        self.dealer = self.draw_hand(self.np_random)
        self.player = self.draw_hand(self.np_random)
        
        # count both my cards and opened dealer one
        self.balance += self.weights[self.dealer[0]] 
        self.balance += self.weights[self.player[0]] 
        self.balance += self.weights[self.player[1]]  
        return self._get_obs()
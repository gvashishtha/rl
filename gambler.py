import random

PROB = 0.4
GAMMA = 1.0
THETA = 0.01
GOAL = 100

# Policy iteration from p. 97 in Sutton and Barto
class PolicyIteration():
    def __init__(self, states, actions, transition_model, gamma):
        self.states = states # list of states
        self.actions = actions # dict mapping states to list of legal actions
        self.values = {} # dict storing mapping from states to values
        self.pi = {} # dict storing a policy (mapping from states to actions)
        for state in states:
            self.values[state] = 0.
            self.pi[state] = random.choice(actions[state])
        self.trans = transition_model # transition model is a mapping from
        # (s, a) pairs to a list of (next, reward, p(next,reward|s,a)) tuples
        self.gamma = gamma # discount factor
        self.last_policy_val = None

    def policy_evaluation(self, theta):
        """
        policy_evaluation, as defined in Sutton and Barto

        Run to determine the values of states (self.values)
        under the current policy (self.pi)
        """
        while True:
            delta = 0.
            for state in self.states:
                v = self.values[state]
                a = self.pi[state]
                self.values[state] = 0.
                for (next, r, p) in self.trans[(state, a)]:
                    self.values[state] += p*(r+self.gamma*self.values[next])
                delta = max(delta, abs(v-self.values[state]))
            if delta < theta:
                break

    def policy_improvement(self, theta):
        """
        policy_improvement

        given a mapping from states to their values, greedily choose a
        new policy based on choosing an action leading to the highest-valued
        successor
        """
        policy_stable = True
        policy_value = 0. # deal with the "subtle bug" in Figure 4.3
        for s in self.states:
            old_action = self.pi[s]
            # compute argmax
            cur_max = 0.
            cur_best_action = old_action
            for a in self.actions[s]:
                value = 0.
                for (next, r, p) in self.trans[(s, a)]:
                    value += p * (r + self.gamma * self.values[next])
                if value > cur_max:
                    cur_best_action = a
                    cur_max = value
            policy_value += cur_max
            self.pi[s] = cur_best_action
            if old_action != self.pi[s]:
                policy_stable = False

        if policy_stable or (self.last_policy_val is not None and
                            policy_value == self.last_policy_val):
        # in case of equality, we k
        # now we are now flipping back and forth between equally good
        # policies
            return (self.values, self.pi)
        else:
            self.last_policy_val = policy_value
            self.policy_evaluation(theta)
            return self.policy_improvement(theta)


# Gambler's problem from p. 100 of Sutton & Barto
states = list(range(0, GOAL+1))
actions = {}
for state in states:
    ub = min(state, GOAL-state)
    if state != GOAL:
        actions[state] = list(range(1,ub+1))
    else:
        actions[state] = [0]
actions[0] = [0]

transition_model = {}
for s in states:
    for a in actions[s]:
        if s + a == GOAL and s != GOAL:
            r = 1
        else:
            r = 0
        transition_model[(s,a)] = [(s+a,r,PROB), (s-a,0,1-PROB)]

Gambler = PolicyIteration(states, actions, transition_model, GAMMA)
Gambler.policy_improvement(THETA)
print('values of states: {} \n\n Policy (amount to bet at a given state): \
      {}'.format(Gambler.values, Gambler.pi)
)

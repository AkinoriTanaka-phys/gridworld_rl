import numpy as np

class Optimizer():
    '''
    Prototype class of Optimizer.

    Attributes
    ----------
    agt : instance
        of Agent class
    '''
    def __init__(self):
        '''
        Do nothing
        '''
        pass

    def set(self, agt):
        '''
        Parameters
        ----------
        agt : instance
            of Agent class
        '''
        self.agt = agt

    def reset(self):
        '''
        Do nothing
        '''
        pass

    def step(self, s, a, rn, sn):
        '''
        Do nothing
        '''
        pass

    def flush(self):
        '''
        Do nothing
        '''
        pass


class ActorCritic(Optimizer):
    '''
    Class of Actor-Critic optimizer.

    Attributes
    ----------
    agt : instance
        of Agent class
    eta_p : float
        Policy update width.
        should be > 0, and small.
    eta_v : float
        Value update width.
        should be > 0, and small.
    gamma : float
        For defining return.
        should be >= 0 and <= 1.
    t : int
        Timestep for an episode of the MDP.

    Notes
    -----
        - agt.policy should be an instance of Softmax.
        - agt.value should be an instance of Value
    '''
    def __init__(self, agt, eta_p, eta_v, gamma):
        '''
        Sets agt as training target.

        Parameters
        ----------
        agt : instance
            of Agent class.
        eta_p : float
            Policy update width.
            should be > 0, and small.
        eta_v : float
            Value update width.
            should be > 0, and small.
        gamma : float
            For defining return.
            should be >= 0 and <= 1.
        '''
        self.eta_p = eta_p
        self.eta_v = eta_v
        self.gamma = gamma
        self.set(agt)
        self.reset()

    def reset(self):
        '''
        Resets t as 0.
        '''
        self.t = 0

    def step(self, s, a, rn, sn):
        '''
        TD update is executed.

        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's state = position [x, y].
        a : int in policy.env.ACTION_SPACE
            Action value taken after s by agt.
        rn : float
            Reward after (s, a).
        sn : numpy.ndarray
            Shape = (2), and is equal to the agent's state after (s, a).
        '''
        TD_error = self.agt.value(s)[0] - (rn + self.gamma*self.agt.value(sn)[0])
        # critic update
        self.agt.value(s)[0] -= self.eta_v*TD_error
        # actor update
        g = -(self.gamma**self.t)*(TD_error)*(self.agt.policy.grad_log_pi(a, s))
        self.agt.policy.get_logit(s)[:] += self.eta_p*g

        self.t += 1

class REINFORCE(Optimizer):
    '''
    Class of REINFORCE optimizer.

    Attributes
    ----------
    agt : instance
        of Agent class
    eta_p : float
        Policy update width.
        should be > 0, and small.
    eta_v : float
        Value (baseline) update width.
        should be > 0, and small.
    gamma : float
        For defining return.
        should be >= 0 and <= 1.

    SAhist : list
        Recording history of (state, action).
    Rhist : list
        recording history of reward.

    Notes
    -----
        - agt.policy should be an instance of Softmax.
        - agt.value should be an instance of Value
    '''
    def __init__(self, agt, eta_p, eta_v, gamma):
        '''
        Sets agt as training target.

        Parameters
        ----------
        agt : instance
            of Agent class.
        eta_p : float
            Policy update width.
            should be > 0, and small.
        eta_v : float
            Value (baseline) update width.
            should be > 0, and small.
        gamma : float
            For defining return.
            should be >= 0 and <= 1.
        '''
        self.gamma = gamma
        self.eta_p = eta_p
        self.eta_v = eta_v
        self.set(agt)
        self.reset()
        
    def reset(self):
        '''
        Resets the histories
            - SAhist
            - Rhist
        '''
        self.SAhist = []
        self.Rhist = [0]
        
    def step(self, s, a, rn, sn):
        ''' 
        Records the histories
            - SAhist + (s, a),
            - Rhist + rn.
        4th parameter sn is ignored.

        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's state = position [x, y].
        a : int in policy.env.ACTION_SPACE
            Action value taken after s by agt.
        rn : float
            Reward after (s, a).
        sn : numpy.ndarray (ignored)
            Shape = (2), and is equal to the agent's state after (s, a).
        '''
        self.SAhist.append((s, a))
        self.Rhist.append(rn)
        
    def flush(self):  
        '''
        Monte Carlo update is executed by using the histories
            - SAhist,
            - Rhist.
        '''     
        if self.SAhist != []: 
            G = 0
            for t in np.arange(len(self.Rhist)-1)[::-1]:
                G = self.gamma*G + self.Rhist[t+1]
                s, a = self.SAhist[t]
                # with baseline
                if self.agt.value is not None:
                    delta = G - self.agt.value(s)[0]
                    self.agt.value(s)[0] += self.eta_v*delta
                # without baseline
                else:
                    delta = G
                self.agt.policy.get_logit(s)[a] += (self.gamma**t)*(self.eta_p)*(delta)*(self.agt.policy.grad_log_pi(a, s)[a])

class MonteCarlo(Optimizer):
    '''
    Class of MonteCarlo optimizer that estimates value by counting.

    Attributes
    ----------
    agt : instance
        of Agent class
    gamma : float
        For defining return.
        should be >= 0 and <= 1.

    Gs : numpy.ndarray
        Shape = (lx, ly, 4), records for returns from (s, a).
    visit : numpy.ndarray
        Shape = (lx, ly, 4), counts visiting (s, a).
    erase_history : bool
        If False, Gs nor visit are not reset zero when reset() called.

    SAhist : list
        Recording history of (state, action).
    Rhist : list
        recording history of reward.

    Notes
    -----
        - agt.policy should be an instance of EpsilonGreedy.
        - agt.value is ignored.
    '''
    def __init__(self, agt, gamma, erase_history=True):
        '''
        Sets agt as training target.
        Initialize Gs and visit as zero arrays.

        Parameters
        ----------
        agt : instance
            of Agent class.
        gamma : float
            For defining return.
            should be >= 0 and <= 1.

        erase_history : bool
            If False, Gs nor visit are not reset zero when reset() called.
        '''
        self.gamma = gamma
        self.erase_history = erase_history
        self.set(agt)
        self.reset()
        self.Gs = np.zeros((self.agt.policy.env.lx, self.agt.policy.env.ly, 4))
        self.visit = np.zeros((self.agt.policy.env.lx, self.agt.policy.env.ly, 4))
        
    def reset(self):
        '''
        Resets the histories
            - SAhist
            - Rhist
        and if erase_history is True,
            - Gs
            - visit.
        '''
        self.SAhist = []
        self.Rhist = [0]
        if self.erase_history:
            self.Gs = np.zeros((self.agt.policy.env.lx, self.agt.policy.env.ly, 4)) 
            self.visit = np.zeros((self.agt.policy.env.lx, self.agt.policy.env.ly, 4)) 
        
    def step(self, s, a, rn, sn):
        ''' 
        Records the histories
            - SAhist + (s, a),
            - Rhist + rn.
        4th parameter sn is ignored.

        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's state = position [x, y].
        a : int in policy.env.ACTION_SPACE
            Action value taken after s by agt.
        rn : float
            Reward after (s, a).
        sn : numpy.ndarray (ignored)
            Shape = (2), and is equal to the agent's state after (s, a).
        '''
        x, y = s
        self.SAhist.append((x, y, a))
        self.Rhist.append(rn)
        
    def flush(self):
        '''
        Monte Carlo update is executed by using the histories
            - SAhist,
            - Rhist.
        '''   
        G = 0
        for t in np.arange(len(self.Rhist)-1)[::-1]:
            G = self.gamma*G + self.Rhist[t+1]
            if not self.SAhist[t] in self.SAhist[:t]: # counting 1st visit only
                x, y, a = self.SAhist[t]
                self.Gs[x, y, a] += G
                self.visit[x, y, a] += 1
                s = np.array([x, y])
                self.agt.policy.param(s)[a] = self.Gs[x, y, a]/self.visit[x, y, a]

class SARSA(Optimizer):
    '''
    Class of SARSA optimizer.

    Attributes
    ----------
    agt : instance
        of Agent class
    eta : float
        Q update width.
        should be > 0, and small.
    gamma : float
        For defining return.
        should be >= 0 and <= 1.

    Notes
    -----
        - agt.policy should be an instance of EpsilonGreedy.
        - agt.value is ignored.
    '''
    def __init__(self, agt, eta, gamma):
        '''
        Sets agt as training target.

        Parameters
        ----------
        agt : instance
            of Agent class.
        eta : float
            Q update width.
            should be > 0, and small.
        gamma : float
            For defining return.
            should be >= 0 and <= 1.
        '''
        self.gamma = gamma
        self.eta = eta
        self.set(agt)
        self.reset()

    def step(self, s, a, rn, sn):
        '''
        TD update is executed.

        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's state = position [x, y].
        a : int in policy.env.ACTION_SPACE
            Action value taken after s by agt.
        rn : float
            Reward after (s, a).
        sn : numpy.ndarray
            Shape = (2), and is equal to the agent's state after (s, a).
        '''
        an = self.agt.play(s) # For simplicity, 2nd action an is sampled in this subroutine.
        TD_error = self.agt.policy.param(s)[a] - (rn + self.gamma*self.agt.policy.param(sn)[an])
        self.agt.policy.param(s)[a] -= self.eta*TD_error

class Qlearning(Optimizer):
    '''
    Class of Q-learning optimizer.

    Attributes
    ----------
    agt : instance
        of Agent class
    eta : float
        Q update width.
        should be > 0, and small.
    gamma : float
        For defining return.
        should be >= 0 and <= 1.

    Notes
    -----
        - agt.policy should be an instance of EpsilonGreedy.
        - agt.value is ignored.
    '''
    def __init__(self, agt, eta, gamma):
        '''
        Sets agt as training target.

        Parameters
        ----------
        agt : instance
            of Agent class.
        eta : float
            Q update width.
            should be > 0, and small.
        gamma : float
            For defining return.
            should be >= 0 and <= 1.
        '''
        self.gamma = gamma
        self.eta = eta
        self.set(agt)
        self.reset()

    def step(self, s, a, rn, sn):
        '''
        TD update is executed.

        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's state = position [x, y].
        a : int in policy.env.ACTION_SPACE
            Action value taken after s by agt.
        rn : float
            Reward after (s, a).
        sn : numpy.ndarray
            Shape = (2), and is equal to the agent's state after (s, a).
        '''
        TD_error = self.agt.policy.param(s)[a] - (rn + self.gamma*np.max(self.agt.policy.param(sn)))
        self.agt.policy.param(s)[a] -= self.eta*TD_error
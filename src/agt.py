import numpy as np

def softmax(theta, axis=0):
    '''
    Paramaters
    ----------
    theta : numpy.ndarray
        Arbitrary shaped input of the softmax function.
    axis : int >= 0
        Indicates axis for interpreting as probability distribution.

    Returns
    ------
    pi : numpy.ndarray
        :math:`\frac{e^{theta[a]}}{\sum_{a} e^{theta[a]}}`

    Examples
    --------
    >>> np.tril(np.ones((3,4)))
    array([[1., 0., 0., 0.],
           [1., 1., 0., 0.],
           [1., 1., 1., 0.]])
    >>> softmax(theta = np.tril(np.ones((3,4))), axis=1)
    array([[0.47536689, 0.1748777 , 0.1748777 , 0.1748777 ],
          [0.36552929, 0.36552929, 0.13447071, 0.13447071],
          [0.29692274, 0.29692274, 0.29692274, 0.10923177]])
    '''
    pi = np.exp(theta)
    pi = pi/np.sum(pi, axis=axis, keepdims=True)
    return pi

####################################################################
class Agent():
    '''
    Class for automatically controlled agent.
    Basic functions:
        - setting it to the target environment,
        - setting the internal structure of the agent, policy, estimated value,
        - choosing action for given state in the target environment,
    are included.

    Attributes
    ----------
    policy : instance 
        of class Policy or its inheritance
    value : instance 
        of class Value

    Notes
    -----
    It may be better to add attribute env...
    env is now included in policy and value as one of their attributes.
    '''
    def __init__(self, env, Policy_class, Value_class=None):
        '''
        Parameters
        ----------
        env : instance 
            of class GridWorld or its inheritance
        Policy_class : class
            Not instance.
            The constructor generates new instance and make it its attribute policy.
        Value_class : class
            Not instance.
            The constructor generates new instance and make it its attribute value.

        See Also
        --------
        Policy : class
            Softmax : class (inheritance of Policy)
            EpsilonGreedy : class (inheritance of Policy)
        Value : class
        '''
        self.policy = Policy_class(env)
        if Value_class is not None:
            self.value = Value_class(env)
        else:
            self.value = None

    def play(self, s):
        '''
        Samples an action using the attribute policy.

        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's current state = position [x, y].

        Returns
        -------
        a : int in policy.env.ACTION_SPACE
        '''
        a = self.policy.sample(s)
        return a

class You():
    '''
    Class for interactive control of the environment by hand.

    Attributes
    ----------
    env : instance 
        of class GridWorld or its inheritance
    '''
    def __init__(self, env):
        '''
        Parameters
        ----------
        env : instance 
            of class GridWorld or its inheritance
        '''
        self.env = env

    def play(self, s):
        '''
        Requires string input in [0,1,2,3].

        Parameters
        ----------
        s : 
            Dummy parameter.     

        Returns
        -------
        a : int in env.ACTION_SPACE
        '''
        while True:
            a = int(input(f"type 0-3 {self.env.ACTION2STR}: "))
            if a in self.env.ACTION_SPACE:
                break
        return a

####################################################################
class Policy():
    '''
    Class for Random policy = uniform distribution.

    Attributes
    ----------
    env : instance 
        of class GridWorld or its inheritance
    param : instance 
        of class TableParam
        includes trainable parameters as a table with shape (lx, ly, 4).

    See Also
    --------
    TableParam : class
    '''
    def __init__(self, env):
        '''
        Parameters
        ----------
        env : instance
            of class GridWorld or its inheritance
        '''
        self.env = env
        self.param = TableParam(env, output_size=4)

    def sample(self, s):
        '''
        Random sample from uniform distribution on env.ACTION_SPACE.

        Parameters
        ----------
        s : 
            Dummy parameter.    

        Returns
        -------
        a : int in env.ACTION_SPACE
        '''
        a = np.random.choice(self.env.ACTION_SPACE)
        return a

class Softmax(Policy):
    '''
    Class for Random policy = softmax distribution.

    Attributes
    ----------
    env : instance 
        of class GridWorld or its inheritance
    param : instance 
        of class TableParam
        includes trainable parameters as a table with shape (lx, ly, 4).
    '''
    def sample(self, s):
        '''
        Random sample from softmax distribution on env.ACTION_SPACE.

        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's current state = position [x, y]. 

        Returns
        -------
        a : int in env.ACTION_SPACE

        See Also
        --------
        get_prob() : method
        '''
        pi_s = self.get_prob(s)
        a = np.random.choice(self.env.ACTION_SPACE, p=pi_s)
        return a

    def get_prob(self, s):
        '''
        Gets :math:`[\pi(a=0|s), \pi(a=1|s), \pi(a=2|s), \pi(a=3|s)]` as numpy.ndarray.
        
        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's current state = position [x, y]. 

        Returns
        -------
        : numpy.ndarray
            Shape = ACTION_SPACE.shape

        See Also
        --------
        get_logit() : method
        softmax() : function
            defined in outside of this class.
        '''
        logit = self.get_logit(s)
        return softmax(logit)

    def get_logit(self, s):
        '''
        Gets logit of :math:`\pi(a|s)`

        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's current state = position [x, y]. 

        Returns
        -------
        : numpy.ndarray
            Shape = ACTION_SPACE.shape

        See Also
        --------
        TableParam : class of param
            The output of this method is implemented by __call__() method of TableParam.
        '''
        return self.param(s)

    def grad_log_pi(self, a, s):
        '''
        Gradient of :math:`\log\pi(a|s)` with respect to theta = param(s).
        More explicitly, it returns 
            :math:`\partial_{theta[a]} [\log\pi(a=0|s),...,\log\pi(a=3|s)]`
        based on softmax policy.
        
        Parameters
        ----------
        a : int in env.ACTION_SPACE
            The value should be in ACTION_SPACE.
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's current state = position [x, y]. 

        Returns
        -------
        : numpy.ndarray
            Shape = ACTION_SPACE.shape

        See Also
        --------
        get_prob() : method
        '''
        return np.identity(len(self.env.ACTION_SPACE))[a] -  self.get_prob(s)

class EpsilonGreedy(Policy):
    '''
    Class for Random policy = epsilon-greedy.

    Attributes
    ----------
    env : instance 
        of class GridWorld or its inheritance
    param : instance 
        of class TableParam
        includes trainable parameters as a table with shape (lx, ly, 4).
    epsilon : float
        Should be in the interval [0, 1] in real values.
        For epsilon-greedy probability.

    See Also
    --------
    TableParam : class
    '''
    def __init__(self, env, epsilon=0.1):
        '''
        Parameters
        ----------
        env : instance 
            of class GridWorld or its inheritance
        epsilon : float
            Should be in the interval [0, 1] in real values.
            For epsilon-greedy probability.

        See Also
        --------
        Policy : class
            The attribute param is defined in the parent class Policy.
        '''
        super().__init__(env)
        self.epsilon = epsilon
    
    def sample(self, s):
        '''
        Returns a by epsilon-greedy policy with estimated action-value param(s).

        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's current state = position [x, y]. 

        Returns
        -------
        a : int in env.ACTION_SPACE
        '''
        Qs = self.param(s) # = [Q(s, 0), Q(s, 1), Q(s, 2), Q(s, 3)]
        if np.random.rand()<1-self.epsilon:
            a = np.random.choice(np.arange(len(Qs))[(Qs == np.max(Qs))]) #np.argmax(Qs)
        else:
            a = np.random.choice(np.arange(len(Qs)))
        return a

####################################################################
class Value():
    '''
    Class for estimating state-value.

    Attributes
    ----------
    env : instance 
        of class GridWorld or its inheritance
    param : instance 
        of class TableParam
        includes trainable parameters as a table with shape (lx, ly, 1).

    See Also
    --------
    TableParam : class
    '''
    def __init__(self, env):
        '''
        Parameters
        ----------
        env : instance 
            of class GridWorld or its inheritance
        '''
        self.env = env
        self.param = TableParam(env, output_size=1)

    def __call__(self, s):
        '''
        Returns
        -------
        param(s) : numpy.ndarray
            Shape = (1), representing estimated state value.
        '''
        return self.param(s)

####################################################################
class TableParam():
    '''
    Class for table-formed parameters.

    Attributes
    ----------
    env : instance 
        of class GridWorld or its inheritance
    table : numpy.ndarray
        Shape = (lx, ly, output_size), and first two components = state.
    output_size : int > 0
        Dimension of the parameter on each position [x, y].
    '''
    def __init__(self, env, output_size=4, init=0):
        '''
        Parameters
        ----------
        env : instance 
            of class GridWorld or its inheritance
        output_size : int > 0
            Dimension of the parameter on each position [x, y].
        init : float
            Initialize each component randomly from uniform(0, init].
        '''
        self.env = env
        self.output_size = output_size
        self.table = init*np.random.rand(env.lx, env.ly, output_size)

    def __call__(self, s):
        '''
        Parameters
        ----------
        s : numpy.ndarray
            Shape = (2), and is equal to the agent's state = position [x, y]. 

        Returns
        -------
        : numpy.ndarray
            Shape = (output_size)
        '''
        x, y = s[0], s[1]
        return self.table[x, y, :]

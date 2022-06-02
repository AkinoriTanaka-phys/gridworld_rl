import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import copy

plt.style.use('fivethirtyeight')

class GridWorld():
    ''' 
    Naive implementation of Grid world.
    Basic functions:
        - controling transition of state and reward,
        - rendering,
    are included. 

    Attributes
    ----------
    lx : int
        Size of x-axis.
    ly : int
        Size of y-axis.
    figsize : int
        Figsize of rendering.
    agt_color : str
        Agent's color in rendering.
    goal_color : str
        Goal's color in rendering.
    
    ACTION_SPACE : numpy.ndarray
        1D array indicating discrite action space.
    ACTION2VECT : dir
        It converts action a in ACTION_SPACE to the corresponding vector.
    ACTION2STR : dir
        It converts action a in ACTION_SPACE to the string indicating direction for move.

    tile : numpy.ndarray
        Boolean array indicating movable coordinates.
        If tile[x, y] is 1, agent can move on [x, y], else if it is 0, agent cannot go through [x, y].
    movable : numpy.ndarray
        List of movable coordinates with shape (-1, 2).
    holes : numpy.ndarray
        List of impassable coordinates with shape (-1, 2).

    goal : numpy.ndarray
        Shape = (2), and indicating goal's coordinate, [x_goal, y_goal].
    state : numpy.ndarray
        Shape = (2), and indicating agent's coordinate, [x_agt, y_agt].
    t : int
        Time steps counted by get_next_state(), resetted by reset().
    status : str
        Carry agent's status as string.
    reward_goal : float
        Reward for reaching to the goal.
    reward_usual : float
        Reward for other cases.
    reward_const : float
        This value is added to reward in each time-step.
    TIMEOUT : int
        If t exceeds it, is_terminated() returns True even if the agent has not been at goal.
    '''
    def __init__(self, lx, ly, figsize=5):
        '''
        Parameters
        ----------
        lx : int
            Size of x-axis.
        ly : int
            Size of y-axis.
        figsize : int
            Figsize of rendering.
        '''
        self.lx, self.ly = lx, ly
        self.figsize = figsize
        self.agt_color = "black"
        self.goal_color = "red"

        self.generate()
        self.reset()

        self.ACTION_SPACE = np.array([0,1,2,3])
        self.ACTION2VECT = { 0: np.array([0, -1]),
                             1: np.array([0, +1]),
                             2: np.array([-1, 0]),
                             3: np.array([+1, 0])}
        self.ACTION2STR = {0:'up', 1:'down', 2:'left', 3:'right'}

        self.status = 'Initialized'
        self.reward_goal = 1
        self.reward_usual = 0
        self.reward_const = 0

        self.TIMEOUT = 100

    def generate(self):
        '''
        Sets all positions are movable, and goal = [0, 0].
        '''
        self.tile = np.ones(shape=(self.lx, self.ly))
        self.load_tile()
        self.goal = np.array([0, 0])
        
    def reset(self, s0=None):
        ''' 
        Resets the position of agent (not goal).
        If the parameter s0 = [x0, y0] is indicated, it puts agent on s0, else it randomly put agent on a movable coordinate.
        
        Parameters
        ----------
        s0 : numpy.ndarray
            Shape = (2), and indicating initial agent's coordinate, [x_agt, y_agt].
            the attribute state is setted by this.

        Returns
        -------
        : numpy.ndarray
            By get_state()

        See Also
        --------
        get_state()
        '''
        if s0 is None:
            floor_labels = np.arange(len(self.movables))
            start_floor_label = np.random.choice(floor_labels)
            s0 = self.movables[start_floor_label]
        self.state = s0
        self.status = 'Reset'
        self.t = 0
        return self.get_state()

    def is_solved(self):
        '''
        Returns
        -------
        : bool
            True if the attribute state = the attribute goal, else False.
        '''
        return self.goal.tolist()==self.state.tolist()

    def is_terminated(self):
        '''
        Returns
        -------
        : bool
            True if self.state = self.goal or the timestep exceeds self.TIMEOUT, else False.
        '''
        return (self.t > self.TIMEOUT) or self.is_solved()
    
    def get_state(self):
        '''
        Returns
        -------
        : numpy.ndarray
            deepcopy of teh attribute state.
        '''
        return copy.deepcopy(self.state)#
            
    def get_next_state(self, a):
        '''
        Makes the attribute state the next coordinate given by action a.

        Parameters
        ----------
        a : int 
            The value should be in ACTION_SPACE.

        Returns
        -------
        : numpy.ndarray
            By get_state()

        See Also
        --------
        get_state()
        '''
        add_vector_np = self.ACTION2VECT[a]
        if (self.state + add_vector_np).tolist() in self.movables.tolist():
            self.state = self.state + add_vector_np
            self.status = 'Moved'
        else:
            self.status = 'Move failed'
        self.t += 1
        return self.get_state()
    
    def get_reward(self):
        '''
        Returns
        -------
        reward + reward_const : float

        See Also
        --------
        is_solved()
        '''
        if self.is_solved():
            reward = self.reward_goal
        else:
            reward = self.reward_usual
        return reward + self.reward_const
    
    def step(self, a):
        '''
        Executes
            - get_next_state(a)
            - get_reward()

        Parameters
        ----------
        a : int 
            The value should be in ACTION_SPACE.
        
        Returns
        -------
        : tuple
            state_next, reward, is_terminated(), {}

        See Also
        --------
        get_next_state() : method
            Makes the attribute state the next coordinate given by action.
        get_reward() : method
            Returns reward.
        '''
        state = self.get_next_state(a)
        reward = self.get_reward()
        return state, reward, self.is_terminated(), {}
        
    def load_tile(self):
        '''
        Generates attributes, 
        - movables and 
        - holes

        See Also
        --------
        GridWorld : class
            The two attributes are explained in help(GridWorld).
        '''
        self.movables = np.array(list(np.where(self.tile==True))).T # (#white tiles, 2), 2 means (x,y) coordinate
        self.holes = np.array(list(np.where(self.tile==True))).T # (#black tiles, 2)
    
    def render_tile(self, ax, cmap='gray'):
        '''
        Renders basic structure of the gridworld, movables and holes.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            main axes for the plot.
        cmap : str
            colormap for the plot.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
        '''
        ax.imshow(self.tile.T, interpolation="none", cmap=cmap, vmin=0, vmax=1)
        return ax
    
    def render_arrows(self, ax, values_table, epsilon=10**(-5)):
        '''
        Renders vector field, case for only len(ACTION_SPACE)=4 is implemented.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            main axes for the plot.
        values_table : numpy.ndarray
            Shape = (lx, ly, 4), indicating vector field.
        epsilon : float (small value)
            Each vector component's strength is plotted by colormap after normalization.
            This parameter is used to avoid overflow dividing by 0 in normalization procedure.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
        '''
        lx, ly, _ = values_table.shape
        vmins = np.min(values_table+epsilon, axis=2).reshape(lx, ly, 1)
        offset = - vmins
        vnoed = values_table*self.tile.reshape(lx, ly, 1) + offset 
        vnoed_maxs = np.max(vnoed, axis=2).reshape(lx, ly, 1)
        vt = np.transpose(vnoed/(vnoed_maxs), (1,0,2))
        width = 0.5
        X, Y= np.meshgrid(np.arange(0, lx, 1), np.arange(0, ly, 1))
        ones = .5*np.ones(lx*ly).reshape(lx, ly)
        zeros= np.zeros(lx*ly).reshape(lx, ly)
        ax.quiver(X, Y, zeros, ones, vt[:,:,0], alpha=0.8, cmap='Reds', scale_units='xy', scale=1) # up
        ax.quiver(X, Y, zeros, -ones, vt[:,:,1], alpha=0.8, cmap='Reds', scale_units='xy', scale=1)# down
        ax.quiver(X, Y, -ones, zeros, vt[:,:,2], alpha=0.8, cmap='Reds', scale_units='xy', scale=1)# left
        ax.quiver(X, Y, ones, zeros, vt[:,:,3], alpha=0.8, cmap='Reds', scale_units='xy', scale=1) # right
        return ax
        
    def render(self, fig=None, ax=None, lines=None, values_table=None, canvas=False):
        '''
        Renders all. 

        Parameters
        ----------
        fig: matplotlib.figure.Figure (optional)
            main figure.
        ax : matplotlib.axes._subplots.AxesSubplot (optional)
            main axes for the plot.
            If None, fig and ax are generated.
        lines : list (optional)
            List of [s_{t}, s_{t+1}] for plotting history of the play.
        values_table : numpy.ndarray (optional)
            Shape = (lx, ly, 4), indicating vector field.
        canvas : bool
            If True, returns ax and not showing the result.

        Returns
        -------
        If canvas is True,
        ax : matplotlib.axes._subplots.AxesSubplot (optional)
            main axes for the plot.
        else
        plt.show()

        See Also
        --------
        render_tile() : method
            Renders basic structure of the gridworld, movables and holes.
        render_arrows() : method
            Renders vector field, case for only len(ACTION_SPACE)=4 is implemented.
        '''
        if ax is None:
            fig = plt.figure(figsize=(self.figsize, self.figsize))
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y')
        ax = self.render_tile(ax)
        if values_table is not None:
            ax = self.render_arrows(ax, values_table)
        if self.goal is not None:
            ax.scatter(self.goal[0], self.goal[1], marker='d', s=100, color=self.goal_color, alpha=0.8, label='goal')
        if self.state is not None:
            ax.scatter(self.state[0], self.state[1], marker='o', s=100, color=self.agt_color, alpha=0.8, label='agent')
        if lines is not None:
            lc = mc.LineCollection(lines, linewidths=2, color='black', alpha=0.5)
            ax.add_collection(lc)   
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', scatterpoints=1)
        ax.set_xticks(np.arange(self.lx))
        ax.set_yticks(np.arange(self.ly))
        ax.grid(color = "black", linestyle = "--", alpha = 0.4)
        if canvas:
            return ax
        else:
            plt.show()
                      
    def play(self, agt, show=True, fig=None, ax=None, canvas=False):
        '''
        Plots playing history by agt.

        Parameters
        ----------
        agt : Agent object

        show : bool
            If True, Render the playing history by agt.
        fig: matplotlib.figure.Figure (optional)
            main figure.
        ax : matplotlib.axes._subplots.AxesSubplot (optional)
            main axes for the plot.
            If None, fig and ax are generated.
        canvas : bool
            If True, returns ax and not showing the result.

        Returns
        -------
        if show
            render(fig=fig, ax=ax, lines=lines, canvas=canvas)
        else
            None
        '''
        lines = []
        while not self.is_terminated():
            s0 = self.get_state()
            a = agt.play(s0)
            self.step(a)
            s1 = self.get_state()
            lines.append([s0, s1])
        if show:
            return self.render(fig=fig, ax=ax, lines=lines, canvas=canvas)

class Maze(GridWorld):
    '''
    Naive implementation of Maze based on GridWorld.
    Basic functions:
        - controling transition of state and reward,
        - rendering,
    are included. 

    Attributes
    ----------
    lx : int
        Size of x-axis.
    ly : int
        Size of y-axis.
    figsize : int
        Figsize of rendering.
    sparseness : float > 0
        If sparseness is small, the resultant maze is dense (difficult).
        Taking 1.1 ~ 1.5 are recommended.

    See Also
    --------
    GridWorld : class
        For other attributes, see the help of parent: help(GridWorld)
    '''
    def __init__(self, lx, ly, figsize=5, sparseness=1.5):
        '''
        Parameters
        ----------
        lx : int
            Size of x-axis.
        ly : int
            Size of y-axis.
        figsize : int
            Figsize of rendering.
        sparseness : float > 0
            If sparseness is small, the resultant maze is dense (difficult).
            Taking 1.1 ~ 1.5 are recommended.

        See Also
        --------
        GridWorld : class
            For other attributes, see the help of parent: help(GridWorld)
        '''
        self.sparseness = sparseness
        super().__init__(lx, ly, figsize)

    def generate(self):
        '''
        Randomly generate the maze.
        '''
        x = np.random.randn(self.lx*self.ly).reshape(self.lx, self.ly)
        y = (x < self.sparseness)*(x > -self.sparseness)
        self.tile = y
        self.load_tile()
        
        floor_labels = np.arange(len(self.movables))
        goal_floor_label = np.random.choice(floor_labels)
        self.goal = self.movables[goal_floor_label]
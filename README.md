# gridworld_rl

Repository for simple implementation of tabular reinforcement learning in grid world with [Gym](https://github.com/openai/gym)-styled API.

## Requirements

- `numpy`
- `matplotlib`
- `copy`

## Usage

Before the training, we need to

1. build environment, e.g. `env = Maze(15, 15, sparseness=1.3)`,
3. make agent, e.g. `agt = Agent(env, Policy_class=Softmax, Value_class=Value)`,
5. set optimizer, e.g. `opt = REINFORCE(agt, eta_p=.5, eta_v=.5, gamma=.99)`.

Once we build `env`, `agt`, `opt`, we can use identical training subroutine like:

```python
N_episode = 1000

for episode in range(N_episode):
    env.reset()
    opt.reset()
    while not env.is_terminated():
        s = env.get_state()
        a = agt.play(s)
        sn, rn, done, _ = env.step(a)
        opt.step(s, a, rn, sn)
    opt.flush() 
```

To check the behavior of the agent schematically, run:

```python
env.reset()
env.play(agt)
```

To visualize policy of the agent, run:

```python
env.render(values_table=agt.policy.param.table)
```

## Class implementations

Environment (src/env.py):
- `Maze`: maze environment

Agents (src/agt.py):
- `You`: control by hand
- `Agent`: autonomous agent based on policy:
  - `Softmax`: tabular softmax policy
  - `EpsilonGreedy`: tabular epsilon-greedy policy

`Agent` can include `Value` to infer state-value function.

Optimizers (src/opt.py):
- for `Softmax` policy:
  - `REINFORCE`
  - `ActorCritic`
- for `EpsilonGreedy` policy:
  - `MonteCarlo`
  - `SARSA`
  - `Qlearning`

And other prototype classes.
See `help()` for more details of each class.

import argparse
import random
from common import DeepQAgent
from ai_safety_gridworlds.environments.tomato_watering import TomatoWateringEnvironment

parser = argparse.ArgumentParser(description='Learning (Hopefully) Safe Agents in Gridworlds')

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
    help='Which to device to use for DQN in PyTorch')

parser.add_argument('--replay-capacity', type=int, default=5000,
    help='Capacity of replay buffer')

parser.add_argument('-L', '--lr', type=float, default=.2,
    help='Learning rate')

parser.add_argument('-E', '--epsilon', type=float, default=.1,
    help='Exploration constant for epsilon greedy policy')

parser.add_argument('--discount', type=float, default=.99,
    help='Agent-death probability complement. x_x')

parser.add_argument('--batch-size', type=int, default=64,
    help='Batch size for Q network updates')

parser.add_argument('--n-layers', type=int, default=3,
    help='Number of (non-input) layers for Q network')

parser.add_argument('--n-hidden', type=int, default=128,
    help='Number of neurons per hidden layer')

args = parser.parse_args()

env = TomatoWateringEnvironment()
print(env.action_spec())

agent = DeepQAgent(env, args)

done = True
for i in range(105):
    if done:
        (step_type, reward, discount, state), done = env.reset(), False

    # Act
    action = agent.act_explore(state)
    step_type, reward, discount, successor = env.step(action)

    # Learn
    agent.learn(state, action, reward, successor)

    # Repeat
    done = step_type.value == 2
    state = successor
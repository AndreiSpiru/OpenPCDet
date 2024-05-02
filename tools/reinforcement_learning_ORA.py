import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define a simple policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

class CustomEnv:
    def __init__(self, examples, k):
        self.examples = examples
        self.k = k
        self.num_examples = len(examples)
        self.current_example_idx = None
        self.current_example = None
        self.current_step = 0

    def reset(self):
        self.current_example_idx = np.random.randint(0, self.num_examples)
        self.current_example = self.examples[self.current_example_idx]
        self.current_step = 0
        return self.current_example

    def step(self, actions):
        selected_rows = self.current_example[actions]
        reward = self.calculate_reward(selected_rows)

        self.current_step += 1
        done = (self.current_step >= len(self.current_example))

        return selected_rows, reward, done

    def calculate_reward(self, selected_rows):
        # Example of a dummy reward function: sum all elements in selected rows
        return np.sum(selected_rows)


def train_reinforce(env, policy_net, optimizer, num_episodes, k=3, gamma=0.99):
    for episode in range(1, num_episodes + 1):
        state = torch.FloatTensor(env.reset())
        saved_log_probs = []
        rewards = []

        while True:
            # Forward pass through policy network to get action probabilities
            action_probs = policy_net(state)
            dist = Categorical(action_probs)
            
            # Sample k actions (indices) from the distribution
            action_indices = dist.sample((k,))
            actions = action_indices.tolist()  # Convert to a list of indices
            #print(f"Episode {episode}, Actions: {actions}")  # Debugging statement

            # Take action in the environment
            next_state, reward, done = env.step(actions)
            rewards.append(reward)
            print(f"Episode {episode}, Reward: {reward}, Done: {done}")  # Debugging statement

            # Save log probability of the selected actions
            log_probs = dist.log_prob(action_indices)
            saved_log_probs.append(log_probs)

            state = torch.FloatTensor(next_state) if not done else None
            if done:
                break

        # Compute discounted rewards-to-go and normalize
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # REINFORCE policy gradient update
        policy_loss = []
        for log_probs, discounted_reward in zip(saved_log_probs, discounted_rewards):
            policy_loss.append(-torch.sum(log_probs) * discounted_reward)

        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} - Total Reward: {sum(rewards)}")

    print("Training completed.")

def main():
    # Initialize policy network and optimizer
    input_size = 2  # Number of features per row
    hidden_size = 64
    k = 3  # Number of rows to select
    output_size = k  # Output size (number of actions)
    
    policy_net = PolicyNetwork(input_size, hidden_size, output_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    # Create custom environment
    num_examples = 300
    examples = [np.random.randn(np.random.randint(5, 10), input_size) for _ in range(num_examples)]
    env = CustomEnv(examples, k)

    # Train the policy network using REINFORCE algorithm
    train_reinforce(env, policy_net, optimizer, num_episodes=1000, k=k)

if __name__ == "__main__":
    main()

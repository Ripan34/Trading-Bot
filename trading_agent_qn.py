from nn import QNetwork, ReplayBuffer
import torch.nn as nn
import random
import torch
import torch.optim as optim
import torch.nn as nn

class TradingAgentQN:
    def __init__(self, env, action_size, state_size, epsilon, batch_size, gama, min_epsilon, epsilon_decay):
        self.env = env
        self.q_network = QNetwork(state_size, action_size)
        self.memory = ReplayBuffer(100000)
        self.target_network = QNetwork(state_size, action_size)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gama 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).view(1, -1)  # Flatten (20, 4) → (1, 80)
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def update(self, episode, target_update_frequency):
        if len(self.memory) > 1000:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # Flatten states: (batch_size, 20, 4) → (batch_size, 80)
            states = torch.FloatTensor(states).view(self.batch_size, -1)
            next_states = torch.FloatTensor(next_states).view(self.batch_size, -1)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)

            q_values = self.q_network(states)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = self.target_network(next_states)
                max_next_q = torch.max(next_q_values, dim=1)[0]
                target_q = rewards + self.gamma * max_next_q * (1 - dones)

            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode % target_update_frequency == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())


    def push_buffer(self, state, action, reward, next_state, done):
        self.memory.push(state.flatten(),
                     action,
                     reward,
                     next_state.flatten(),
                     done)
    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_model(self):
        torch.save(self.q_network.state_dict(), 'trading_q_network.pth')
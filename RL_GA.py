import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load dataset (replace with actual data loading in practice)
df = pd.read_csv("main_dataset.csv")
X = df.drop(columns=['class'])
y = df['class']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Genetic Algorithm Parameters
POP_SIZE = 20  # Number of feature subsets
GENS = 200  # Number of generations
MUT_RATE = 0.2  # Mutation rate

# Initialize Population (Random Feature Selection)
def init_population(size, num_features):
    return [np.random.randint(0, 2, num_features).tolist() for _ in range(size)]

# Fitness Function (Model Performance)
def fitness(chromosome):
    selected_features = np.where(np.array(chromosome) == 1)[0]
    if len(selected_features) == 0:
        return 0  # Avoid empty feature sets
    
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    return accuracy_score(y_test, y_pred)

# RL Environment for Feature Selection
class FeatureSelectionEnv(gym.Env):
    def __init__(self, num_features):
        super(FeatureSelectionEnv, self).__init__()
        self.num_features = num_features
        self.action_space = spaces.Discrete(num_features)  # Select/deselect features
        self.observation_space = spaces.MultiBinary(num_features)
        self.state = np.random.randint(0, 2, num_features)
    
    def step(self, action):
        self.state[action] = 1 - self.state[action]  # Toggle feature
        reward = fitness(self.state)
        return self.state, reward, False, {}
    
    def reset(self):
        self.state = np.random.randint(0, 2, self.num_features)
        return self.state

# Define RL Agent (DQN)
class DQN(nn.Module):
    def __init__(self, num_features):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_features)
        )
    
    def forward(self, x):
        return self.fc(x)

# Train RL Agent
def train_rl_agent(env, episodes=1000, lr=0.001):
    model = DQN(env.num_features)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for episode in range(episodes):
        state = env.reset()
        state_tensor = torch.FloatTensor(state)
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()
        
        next_state, reward, _, _ = env.step(action)
        next_state_tensor = torch.FloatTensor(next_state)
        target = reward + 0.99 * torch.max(model(next_state_tensor)).item()
        loss = criterion(q_values[action], torch.tensor(target))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward:.4f}")
    
    return model

# RL-Guided Mutation
def rl_mutate(chromosome, rl_model, env):
    state_tensor = torch.FloatTensor(chromosome)
    action = torch.argmax(rl_model(state_tensor)).item()  # RL decides which feature to toggle
    chromosome[action] = 1 - chromosome[action]  # Toggle the selected feature
    return chromosome

# GA with RL-guided mutation
def genetic_algorithm_with_rl(rl_model):
    population = init_population(POP_SIZE, X_train.shape[1])
    
    for gen in range(GENS):
        selected = sorted(population, key=fitness, reverse=True)[:POP_SIZE//2]  # Selection
        new_population = []
        
        while len(new_population) < POP_SIZE:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = parent1[:], parent2[:]
            
            # Use RL-guided mutation
            child1 = rl_mutate(child1, rl_model, env)
            child2 = rl_mutate(child2, rl_model, env)
            
            new_population.extend([child1, child2])
        
        population = new_population[:POP_SIZE]
    
    best_chromosome = max(population, key=fitness)
    return best_chromosome

# Train RL Agent
env = FeatureSelectionEnv(X_train.shape[1])
rl_model = train_rl_agent(env)

# Run RL-GA Hybrid
best_feature_subset = genetic_algorithm_with_rl(rl_model)
print("Final RL-GA Selected Features:", best_feature_subset)

# Evaluate Final Feature Subset with Logistic Regression
selected_features = np.where(np.array(best_feature_subset) == 1)[0]
if len(selected_features) > 0:
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
else:
    print("No features were selected by RL-GA.")
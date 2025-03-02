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
    return recall_score(y_test, y_pred)

# Genetic Algorithm
def genetic_algorithm():
    population = init_population(POP_SIZE, X_train.shape[1])
    
    for gen in range(GENS):
        print(f"Generation {gen}")
        scores = [(fitness(chromosome), chromosome) for chromosome in population]
        scores.sort(reverse=True, key=lambda x: x[0])
        population = [chromosome for _, chromosome in scores[:POP_SIZE // 2]]  # Select top individuals
        
        # Crossover
        offspring = []
        while len(offspring) < POP_SIZE // 2:
            p1, p2 = random.sample(population, 2)
            crossover_point = random.randint(1, len(p1) - 1)
            child = p1[:crossover_point] + p2[crossover_point:]
            offspring.append(child)
        
        # Mutation
        for child in offspring:
            if random.random() < MUT_RATE:
                mutate_idx = random.randint(0, len(child) - 1)
                child[mutate_idx] = 1 - child[mutate_idx]
        
        population.extend(offspring)
    
    best_solution = max(population, key=fitness)
    return best_solution

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

# Run GA
best_feature_subset = genetic_algorithm()
print("Final Selected Features:", best_feature_subset)

# Train RL agent
env = FeatureSelectionEnv(X_train.shape[1])
rl_model = train_rl_agent(env)

# Get best feature subset from RL
state = env.reset()
for _ in range(X_train.shape[1]):
    state_tensor = torch.FloatTensor(state)
    action = torch.argmax(rl_model(state_tensor)).item()
    state, _, _, _ = env.step(action)

print("Final RL-Selected Features:", state)

# Evaluate Final Feature Subset with Logistic Regression
selected_features = np.where(np.array(state) == 1)[0]
if len(selected_features) > 0:
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred)) 
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))

else:
    print("No features were selected by RL.")
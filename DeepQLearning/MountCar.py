import gym
import time
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam

"""
Reminder:
    traditional Q-learning:
        Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
    DQN:
        target = reward(s,a) + gamma * max(Q(s')
"""


# DQN is used to represent the agent
class DQN:
    def __init__(self, env, eps=1.0, minEps=0.01, epsDecay=0.995, gamma=0.85, alpha=0.1):
        self.env = env              # Stores the environment the agent is in
        self.eps = eps              # Stores the exploration chance
        self.minEps = minEps        # Stores the minimum exploration chance
        self.epsDecay = epsDecay    # Stores the rate of decay of exploration

        # Variables needed for DQN equation
        self.gamma = gamma
        self.alpha = alpha

        self.memory = []            # Stores the previous actions of the agent

        self.model = self.createNetwork()           # Stores the model
        self.targetModel = self.createNetwork()     # Stores the target model

    # Build the network
    def createNetwork(self):
        # Add all of the layers to the network
        model = Sequential()

        stateShape = self.env.observation_space.shape
        model.add(InputLayer(input_shape=stateShape))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mse",
                      optimizer=Adam(),
                      metrics=['mae'])
        return model

    # Add the state to the agents memory
    def remember(self, curState, action, reward, newState, done):
        self.memory.append([curState, action, reward, newState, done])

    # Update the model using the memory
    def learn(self):
        # Only update when the memory is large enough
        batchSize = 32
        if len(self.memory) < batchSize:
            return

        # Select random states from memory
        samples = random.sample(self.memory, batchSize)
        for sample in samples:
            # Update the model probabilities using equation
            state, action, reward, newState, done = sample
            target = self.targetModel.predict(state)
            # Give reward if model succeeded
            if done:
                target[0][action] = reward
            else:
                qFuture = max(self.targetModel.predict(newState)[0])
                qValue = self.alpha * (reward + self.gamma * qFuture)
                target[0][action] = qValue
            self.model.fit(state, target, epochs=1, verbose=0)

    # Update target model
    def trainTarget(self):
        weightMatrix = []
        # Get the weights
        for layer in self.model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)

        # Set the weights
        for i, layer in enumerate(self.targetModel.layers):
            weights = weightMatrix[i]
            layer.set_weights(weights)

    # Return the best action based on model prediction
    def getBestAction(self, state):
        return np.argmax(self.model.predict(state)[0])

    # Move the agent
    def act(self, state):
        # Update epsilon
        self.eps *= self.epsDecay
        self.eps = max(self.eps, self.minEps)

        # Check for exploration
        if np.random.random() < self.eps:
            return self.env.action_space.sample()

        # Get best move
        return self.getBestAction(state)

    # Save the model
    def saveModel(self, fn):
        self.model.save(fn)


# Run the simulation
def main(env, agent):
    epochs = 10000              # Number of attempts
    numCompletions = 0          # Number of successful attempts
    targetTrainCount = 50       # Frequency of print statements

    startTime = time.time()

    # Simulate each epoch
    for i in range(epochs):
        curState = env.reset().reshape(1, 2)
        stepCount = 0
        done = False

        # Loop until reached goal
        while not done:
            # Move agent
            action = agent.act(curState)
            env.render()
            newState, reward, done, _ = env.step(action)

            newState = newState.reshape(1, 2)

            if newState[0, 0] >= 0.5:
                reward = 20

            # Update memory
            agent.remember(curState, action, reward, newState, done)
            agent.learn()

            curState = newState
            stepCount += 1

        # Stop trying after too many moves in an attempt
        if stepCount >= 200:
            print('Failed to complete in {} steps - epoch {}'.format(stepCount, i+1))
        else:
            numCompletions += 1
            print('Completed in {} steps - epoch {} - {} completions'.format(stepCount, i+1, numCompletions))
            agent.trainTarget()

        # Tell user how long training is taking
        if i % targetTrainCount == 0:
            print('Time Taken: {}s'.format(np.round(time.time() - startTime, decimals=2)))
            startTime = time.time()
            agent.trainTarget()

        # When completed 25 times save the model
        if numCompletions >= 25:
            agent.save('success.model')
            break


# Run the program
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = DQN(env)
    main(env, agent)

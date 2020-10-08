import numpy as np

# Constant for the size of the board
SIZE = 3


# Environment is used to represent the board during the game
class Environment:
    def __init__(self):
        self.board = np.zeros((SIZE, SIZE))         # The initial board
        self.o = 1.0                                # The o representation
        self.x = -1.0                               # The x representation
        self.numStates = 3**(SIZE*SIZE)             # The number of possible states for the board

    # Used to print the board neatly
    def __str__(self):
        count = 0

        print()
        # Loop through each row
        for i in reversed(self.board):
            # Replace numbers with X and O
            print(' {} | {} | {}'.format(str(i[0]).replace('0.0', ' ').replace(str(self.x), 'X').replace(str(self.o), 'O'),
                                         str(i[1]).replace('0.0', ' ').replace(str(self.x), 'X').replace(str(self.o), 'O'),
                                         str(i[2]).replace('0.0', ' ').replace(str(self.x), 'X').replace(str(self.o), 'O')))

            # Add horizontal border after each row
            if count < 2:
                print('---+---+---')
            count += 1

        return ''

    # Reward the AI when it wins
    def reward(self, sym):
        return 1 if self.isWinner() == sym else 0

    # Hash the state of the board so it can be used by the AI
    def getHashedState(self):
        power = 0
        hashed = 0

        # Loop through each cell
        for i in range(SIZE):
            for j in range(SIZE):
                if self.board[i, j] == self.x:
                    hashed += (3**power) * 2
                else:
                    hashed += (3**power) * self.board[i, j]
                power += 1

        return int(hashed)

    # Check if there is a winner once the game is over
    def isWinner(self):
        # Check each symbol
        for sym in (self.o, self.x):
            # Check each row
            for row in self.board:
                if np.sum(row) == sym*3:
                    return sym

            # Check each column
            for col in self.board.T:
                if np.sum(col) == sym*3:
                    return sym

            # Check leading diagonal
            if sum(self.board.diagonal()) == sym*3:
                return sym

            # Check other diagonal
            if sum(np.fliplr(self.board).diagonal()) == sym*3:
                return sym

        return None

    # Check if the game ended in a draw
    def isDraw(self):
        return not (np.all(self.board) == 0)

    # Check if the game has ended
    def gameOver(self):
        winner = self.isWinner()
        if winner is not None or self.isDraw():
            return True

        return False


# Human is used to represent a human player
class Human:
    def __init__(self, sym):
        self.symbol = sym       # The symbol being used by the player

    # Get the users input and update the board
    def makeMove(self, env):
        # Display the board
        print(env)

        # Get the users input
        move = input('\nMove:\t')

        # Check the move is valid
        possibleMoves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]    # Maps the users input to board coordinates
        while not move.isdigit() or int(move)-1 not in range(SIZE*SIZE) or env.board[possibleMoves[int(move)-1]] != 0:
            move = input('\nInvalid Move!\nMove:\t')

        # Make the users move
        move = int(move) - 1
        env.board[possibleMoves[move]] = self.symbol

    # Allows Human to be substituted for Agent without errors
    def updateStateHistory(self, s):
        pass

    # Allows Human to be substituted for Agent without errors
    def update(self, s):
        pass


# Agent is used to represent an AI player
class Agent:
    def __init__(self, sym, eps=0.1, gamma=0.9, alpha=0.5, v=False):
        self.symbol = sym               # Stores the symbol of the player

        # Used to store variables of model
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.valueTable = []
        self.stateHistory = []
        self.verbose = v

    # Generate the AIs move
    def makeMove(self, env):
        nextMove = None

        # Add chance to make worse move
        if np.random.random() < self.eps:
            possibleMoves = []

            # Get all empty cells
            for i in range(SIZE):
                for j in range(SIZE):
                    if env.board[i, j] == 0:
                        possibleMoves.append((i, j))

            # Pick move randomly
            nextMove = possibleMoves[np.random.choice(len(possibleMoves))]

        else:
            # Generate best move using model
            bestMove = -1
            pos2value = {}

            # Get all empty cells
            for i in range(SIZE):
                for j in range(SIZE):
                    if env.board[i, j] == 0:
                        # Try each move and predict the result
                        env.board[i, j] = self.symbol
                        state = env.getHashedState()
                        env.board[i, j] = 0
                        pos2value[(i, j)] = self.valueTable[state]

                        # Save the best possible move
                        if self.valueTable[state] > bestMove:
                            bestMove = self.valueTable[state]
                            nextMove = (i, j)

            # If the AI is verbose display the board with values of each move
            if self.verbose:
                for i in range(SIZE):
                    print('------------------')
                    for j in range(SIZE):
                        if env.board[i, j] == 0:
                            print(' %.2f|' % pos2value[(i, j)], end='')
                        else:
                            print('  ', end='')
                            if env.board[i, j] == env.x:
                                print('x  |', end='')
                            elif env.board[i, j] == env.o:
                                print('o  |', end='')
                            else:
                                print('   |', end='')
                    print()
                print('------------------')

        env.board[nextMove] = self.symbol

    # Set up the value table
    def initV(self, env, triples):
        self.valueTable = np.zeros(env.numStates)

        # Loop through all possible final states
        for state, winner, ended in triples:
            if ended:
                if winner == self.symbol:
                    self.valueTable[state] = 1          # 1 if win
                else:
                    self.valueTable[state] = 0          # 0 if loss
            else:
                self.valueTable[state] = 0.5            # 0.5 if draw

    # Add the state to the state history
    def updateStateHistory(self, s):
        self.stateHistory.append(s)

    # Update the model
    def update(self, env):
        # Get the reward from the outcome
        reward = env.reward(self.symbol)
        target = reward

        # Update state history
        for prevReward in reversed(self.stateHistory):
            value = self.valueTable[prevReward] + self.alpha * (target - self.valueTable[prevReward])
            self.valueTable[prevReward] = value
            target = value

        self.stateHistory = []


# Generate state winner triples
def getStateWinnerTriples(env, i=0, j=0):
    triples = []

    # Place each value in the given cell
    for v in (0, env.o, env.x):
        env.board[i, j] = v
        if j == 2:
            if i == 2:      # The last cell to be filled (bottom right)
                # Add the state to the triples
                state = env.getHashedState()
                winner = env.isWinner()
                ended = env.gameOver()
                triples.append((state, winner, ended))
            else:
                # Recursively get the triple for next cell horizontally
                triples += getStateWinnerTriples(env, i+1, 0)
        else:
            # Recursively get the triple for next cell vertically
            triples += getStateWinnerTriples(env, i, j+1)

    return triples


# Run the game
def playGame(env, p1, p2):
    currentPlayer = None

    while not env.gameOver():
        # Alternate between the players
        if currentPlayer == p1:
            currentPlayer = p2
        else:
            currentPlayer = p1

        # Make the players move
        currentPlayer.makeMove(env)

        # Update the AI state history
        state = env.getHashedState()
        p1.updateStateHistory(state)
        p2.updateStateHistory(state)

    # Update the AI model
    p1.update(env)
    p2.update(env)

    return env, env.isWinner()


if __name__ == '__main__':
    # Set up environment
    env = Environment()

    # Set up players
    p1 = Agent(env.x)
    p2 = Agent(env.o)

    stateWinnerTriples = getStateWinnerTriples(env)

    p1.initV(env, stateWinnerTriples)
    p2.initV(env, stateWinnerTriples)

    # Train the AIs
    T = 100000
    for t in range(T):
        if t % 200 == 0:
            print(t)
        playGame(Environment(), p1, p2)

    p1.verbose = True
    p1.eps = 0

    human = Human(env.o)

    players = [human, p1]

    # Run game AI vs. Human
    while True:
        # Pick a random player to start
        r = np.random.randint(2)
        env, winner = playGame(Environment(), players[r], players[1-r])

        print(env)

        # Print end game message
        if winner == human.symbol:
            print('\n!!! YOU WIN !!!')
        elif winner in (p1.symbol, p2.symbol):
            print('\n!!! YOU LOSE !!!')
        else:
            print('\n!!! DRAW !!!')

        # Check if user wants to play another game
        if input('\nPlay Again? [Y/n]\n').lower() == 'n':
            break

# Setup

The code divides the work into 2 parts. One is handled by the environment while
the other is handled by the algorithms provided for the problem statement.
Environment is depicted using the *Environment* class. The Environment class
contains methods such as

**reset** - To get the starting state

**performAction** - To get the reward, next state after taking the current state
                    and an action as parameters.

We define the Q function using a tabular form. Every problem uses episodes to learn
the Q function. The algorithm used in both the problems is *epsilon-greedy*. The
epsilon values decrease according to the episode number.

Finally, after the training is over, the state is reset and run once more till we
reach an end. This provides us the final answer.

# Problems

Edit Distance, Longest Path

## Edit Distance

### Problem Statement

Given two strings str1 and str2 and operations(of replace, remove, insert) that can performed on str1.
Find minimum number of edits (operations) required to convert ‘str1’ into ‘str2’.

### Actions

Four actions are possible. First action means no changes are to be done in the
current indices and we move on to the next index of both the strings. Second action
means the current index of the first string is removed. Third action states that
a new alphabet is inserted in the current index of the first string. Fourth action
states that the current index of the first string is replaced with another alphabet.

### Rewards

The rewards are mentioned in the performAction method. The rewards are decided in
a way that it helps the algorithm in guiding the Q function in the right direction.
The reward tries to penalise the edits and hence forces the algorithms to reduce
the overall edits.

### Final run

We reset the state. Get the actions that maximise the Q for that state. Continue
till we reach some end state. The total actions are calculated. Action 0 is not
counted as it doesn't have any edits included. The sum is the minimum edits
required to convert the string because each action(except 0) contains an edit.

## Longest Path

Given a n*n matrix where all numbers are distinct, find the maximum length
path (starting from any cell) such that all cells along the path are in increasing
order with a difference of 1.

### Actions
Four actions are possible. First action means moving up. Second action means moving
down. Third action means moving left while fourth means moving right.

### Rewards

The rewards help ensure that the actions do not make the index move to some illegal
states. They also help ensure that the states adjacent to states with a value difference
of 1 are given more reward.

### Final Run

We just take the state that gives us the maximum Q value across its action. Since
the Q here means how good a state is if we start the sequence from there with a given
action, this final inference helps us get the state that is best to start such sequence.

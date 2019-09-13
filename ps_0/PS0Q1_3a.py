#!/usr/bin/python3

import numpy as np

def all_outcomes(N):
    """Simulate the outcomes of a roll of a six-sided die over 
    N trials using np.random.rand function
    Parameters
    ----------
    N : int
        Number of trials of the die
    Returns
    -------
    outcomes: np.array
        Array of outcomes of die-roll over N trials
    """
    SIDES_IN_DIE = 6
    prob_of_outcomes = np.random.rand(N, SIDES_IN_DIE)
    outcomes = []
    for roll_event in prob_of_outcomes:
        number_rolled = np.argmax(roll_event) + 1
        outcomes.append(number_rolled)
    return np.array(outcomes)

def test_all_outcomes():
    """Function to test all_outcomes() function"""
    N = 1
    outcomes = all_outcomes(N)
    print ("N: {}, outcomes: {}".format(N, outcomes))
    N = 2
    outcomes = all_outcomes(N)
    print ("N: {}, outcomes: {}".format(N, outcomes))
    N = 4
    outcomes = all_outcomes(N)
    print ("N: {}, outcomes: {}".format(N, outcomes))

if __name__ == '__main__':
    test_all_outcomes()
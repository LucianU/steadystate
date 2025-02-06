# steadystate
A small project that analyzes the convergence of several Markov Chains.

We first look at 2 chains with significantly different transition matrices
[`convergence.py`].
Then, we add 3 more chains also with peculiar matrices and see how they converge
to their stationary distribution [`extra_transitions.py`].

As a bonus, we also compare the performance of the same transition matrices on
different initial distributions[`extra_initial_dists.py`].

Finally, we perform simulations with the same matrices to see how things look
empirically[`simulation.py`].

Part of my class in Bayesian Networks and Hidden Markov Models.

Built together with my colleague Mara Fodor.


# Setup
## Poetry (Recommended)
The project uses [Poetry](https://python-poetry.org/) for dependency management.

After installing Poetry:

- install project dependencies:

        poetry install

- activate the project environment:

        poetry shell

- run the different files:

        python -m steadystate.convergence
        python -m steadystate.extra_transitions
        python -m steadystate.extra_initial_dists
        python -m steadystate.simulation


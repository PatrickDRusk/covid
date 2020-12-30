# covid

This repo has a couple notebooks with my analysis of COVID data.

The principal sources of data are the very excellent NY Times data
(https://github.com/nytimes/covid-19-data) and the COVID Tracking
Project data (https://covidtracking.com/data/). It is also supplemented
by some of my home state Massachusetts' data, as described in the
notebook.

The notebooks attempt to do the following:
1. Correct for anomalies in death reporting by the states.
1. Appropriately smooth the data.
1. Slice and dice by various metrics (region, politics, etc.).
1. Estimate total infections.

The better of the notebooks is `calc_state_infections.pdr.ipynb`, since
it uses the infection estimation code. Note that the `pdr` in the name
are my initials. If you wish to submit a PR against a notebook, please
create a version with your own initials to avoid conflicts.

"""A3I analysis/plotting library.

Layered so every consumer reads one shared model:
  load.py      -- discover result files -> one tidy long frame (the grain)
  aggregate.py -- the runs->queries reductions, oracle join, comparison guards
The rendering layers (render.py / figures.py) pull in matplotlib; the analysis
core here does not.
"""

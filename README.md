# fairfs

# Data
Input datasets are not stored directly on github, since they are publicly available from the [UCI ML repository](https://archive.ics.uci.edu/ml/index.php). For further information, see the README in the 'data' folder.

# Running the code
Required libraries can be installed with `pip install requirements.txt`. Once datasets have been downloaded locally, experiments are run with `python fairfs.py`. The code currently runs only one dataset at a time (lines 49-43). Ensure that the correct protected group is selected for the appropriate dataset (line 13), or the code will fail to run. Results will be written to a file called `fairfs_results.csv`.

Graphs presented in the AIES paper can be created by running `python graph_creation.py`. Filepaths should be provided to this module via the list on line 10.

# CZ4032 Data Analytics and Mining

// Best result so far:
Output file:        out6.csv
Private score:      0.214137
Public score:       0.219092

// How to use the bash scripts:
The scripts are for automating the model run.
In order to run a bash script, type "bash [insert script name]" in command line.

- "bash run.sh": run a normal prediction, considering 14 suppliers.
- "bash run1.sh": run multiple prediction by removing 1 supplier at a time.
- "bash run2.sh": run multiple prediction by removing 2 suppliers at a time.

// Number of runs per script:
- run.sh: 1 single run
- run1.sh: 14 runs
- run2.sh: 14 choose 2 = 91 runs

// Updates:
#1 The script run1 is extended for 2 suppliers removal. The run2 is added.
#2 Resolved problems with iterations. Previous output files fewer than 91.
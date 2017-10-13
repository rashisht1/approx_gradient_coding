# No. of workers
N_PROCS=51

# No. of stragglers in our coding schemes
N_STRAGGLERS = 5

# Path to folder containing the data folders
DATA_FOLDER=/straggdata/

IS_REAL = 1

DATASET = amazon-dataset
N_ROWS=26210
N_COLS=241915

# Note that DATASET is automatically set to artificial-data/ (n_rows)x(n_cols)/... if IS_REAL is set to 0 \
 or artificial-data/partial/ (n_rows)x(n_cols)/... if PARTIAL_CODED is also set to 1

generate_random_data:
	python ./src/generate_data.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(N_STRAGGLERS)

arrange_real_data:
	python ./src/arrange_real_data.py $(N_PROCS) $(DATA_FOLDER) $(DATASET) $(N_STRAGGLERS)

naive:   
	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 0 $(N_STRAGGLERS) 0

cyccoded:
	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 0

approxcoded:
	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 1

avoidstragg:
	mpirun -np $(N_PROCS) python main.py $(N_PROCS) $(N_ROWS) $(N_COLS) $(DATA_FOLDER) $(IS_REAL) $(DATASET) 1 $(N_STRAGGLERS) 2
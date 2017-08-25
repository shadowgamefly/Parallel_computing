// This program simulates the flow of heat through a two-dimensional plate.
// The number of grid cells used to model the plate as well as the number of
//  iterations to simulate can be specified on the command-line as follows:
//  ./heated_plate_sequential <columns> <rows> <iterations>
// For example, to execute with a 500 x 500 grid for 250 iterations, use:
//  ./heated_plate_sequential 500 500 250

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

// Define the immutable boundary conditions and the inital cell value
#define TOP_BOUNDARY_VALUE 0.0
#define BOTTOM_BOUNDARY_VALUE 100.0
#define LEFT_BOUNDARY_VALUE 0.0
#define RIGHT_BOUNDARY_VALUE 100.0
#define INITIAL_CELL_VALUE 50.0
#define hotSpotRow 4500
#define hotSpotCol 6500
#define hotSpotTemp 1000


// Function prototypes, comments above only give a brief description,
// detailed explanation and implementation can be found after main method

// directly print the values of the cells given a block, used only for debug purposes
void printMaps(float **cells, int n_x, int n_y,int rank, int type);

// initialize all the cells inside the block with value 50
void initialize_cells(float **cells, int n_x, int n_y, int ghost);

// create a snapshot as ppm file given a cell block
void create_snapshot(float **cells, int n_x, int n_y, int id);

// set the boundary value for a given block for a single node
void set(float ***cells, int n_x, int n_y, int ghost, int rank, int nodes);

// allocate a 2-D array with given number of rows and columns, this 2-D array is
// actually continuous in memory
float **allocate_cells(int n_x, int n_y);

// kill the program
void die(const char *error);

// set the hotspot into correct node and position
void hotspot(float **cells, int n_x, int n_y, int ghost, int rank, int nodes);

// deal with message passing through different node
void send(float **cells, int ghostSize, int ghost, int numrows, int rank, int nodes);

// main method
int main(int argc, char **argv) {
	// Record the start time of the program
	time_t start_time = time(NULL);
	// variable declaration for various iterator in for loop
	int x, y, i, iter;
	// Extract the input parameters from the command line arguments
	// Number of columns in the grid (default = 1,000)
	int real_cols = (argc > 1) ? atoi(argv[1]) : 1000;

	// Number of rows in the grid (default = 1,000)
	int real_rows = (argc > 2) ? atoi(argv[2]) : 1000;

	// the number of inner loop iterations to compute per cell formed from the chunk divisions.
	int iters_per_cell = (argc > 3) ? atoi(argv[3]) : 1000;

	// number of iterations to output the matrix to snapshot.X in which X is the iteration number.
	int iterations_per_snapshot = (argc > 4) ? atoi(argv[4]) : 1000;

	// Number of iterations to perform (default = 100)
	int iterations = (argc > 5) ? atoi(argv[5]) : 100;

	// boundary_thickness number of ghost cell layers to send at a time and how many internal
	// iterations to perform per communication (defaut = 1)
	int ghost = (argc > 6) ? atoi(argv[6]) : 1;

	//initialize MPI
	MPI_Init(&argc, &argv);
	//retrive number of nodes in the whole program
	int nodes;
	MPI_Comm_size(MPI_COMM_WORLD, &nodes);
	// retrive the rank for the current process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// pointer to the cell block, one for current block, one for next
	// we use them alternatively for computaion
	float **cells[2];

	// pointer to a whole final image, used for final image printout
	float **final_result;

	// num_cols and num_rows represents the number of rows/columns
	// for a single node that it need to handle
	int num_cols = real_cols;
  int num_rows = real_rows/nodes;

	// total_rows/cols represents the number of rows/cols that one node need to
	// store, the difference between total vs num comes from the fact that a node
	// need to store extra ghost cells, and fixed place holder for some cells
	int total_rows = num_rows + 2 * ghost;
	int total_cols = num_cols + 2;

	// transferSize means the number of cells every subnode need to tranfer to the
	// main node to create the final graph
	int transferSize = (num_cols + 2) * num_rows;

	// ghostSize means the numebr of cells every subnode need to tranfer between
	// each other for every message passing along computation
	int ghostSize = ghost * total_cols;

	// since for a given number of ghost-cell size say g, a single node can get
	// correct output for the cells for g iterations, so we only need to do
	// iteraions/ghost number of message passing, and after each message passing
	// we can iterate g times
	int transfer = iterations / ghost; // assume always return an int

	// allocate both blocks based on total_rows/cols
	cells[0] = allocate_cells(total_cols, total_rows);
	cells[1] = allocate_cells(total_cols, total_rows);

	// allocate the final block only on master node, note that since each
	// row have two extra fixed placeholder, it also need to be transfered
	if (rank == 0) final_result = allocate_cells(real_cols + 2, real_rows);

	// specify the current/next cell block
	int cur_cells_index = 0, next_cells_index = 1;

	// Initialize the interior (non-boundary) cells to their initial value.
	// Note that we only need to initialize the array for the current time
	// step, since we will write to the array for the next time step
	// during the first iteration.
	initialize_cells(cells[0], num_cols, num_rows, ghost);

	//conter for the iterations, used for determine if we need to
	//create a output image based on iterations_per_snapshot
	int current_iterations = 0;

	//outer fo loop for each message passing
	for (int m = 0; m < transfer ; m++) {
		// message passing between nodes, detail can be found below in the method
		send(cells[cur_cells_index], ghostSize, ghost, num_rows, rank, nodes);

		// as discussed above, for a single message passing, the output for a given
		// node is correct for `ghost` times
		for (i = 0; i < ghost; i++) {
			// every time after calculation reset the boundary value and the hotspot
			set(cells, num_cols, num_rows, ghost, rank, nodes);
			// determine if we have a hotspot based on the size of the plate
			if ((real_cols > hotSpotCol) && (real_rows > hotSpotRow)) {
				// set the hotspot onto the correct node
				hotspot(cells[cur_cells_index], num_cols, num_rows, ghost, rank, nodes);
			}
			// Traverse the plate, computing the new value of each cell
			for (x = 1; x < total_rows - 1 ; x++) {
				for (y = 1; y < total_cols - 1; y++) {
					// iterate a calculation multiple times to increase the granularity
					for (iter = 0; iter < iters_per_cell; iter++) {
					//The new value of this cell is the average of the old values of this cell's four neighbors
						cells[next_cells_index][x][y] = (cells[cur_cells_index][x - 1][y]  +
				                 	        	         cells[cur_cells_index][x + 1][y]  +
				                        	        	 cells[cur_cells_index][x][y - 1]  +
				                                 		 cells[cur_cells_index][x][y + 1]) * 0.25;
					}
				}
			}
			// increase the counter
			current_iterations += 1;

			// if current_iterations have reached the number of iterations to print a snapshot
			// goto the create process
			if (current_iterations % iterations_per_snapshot == 0) goto createSnap;

			// if the process hasn't finished place to continue
			cont:

			// Swap the two arrays
			cur_cells_index = next_cells_index;
			next_cells_index = !cur_cells_index;
		}
	}

	// the process to create the snapshot based on current values calculated
	createSnap:
	//tag 10 for all the transfer
	//transfer the all the calculated result to the node 0 (despite the ghost cells)

	// node 0
  if (rank == 0) {
		// transfer its own value to the front of the final map
		memcpy(final_result[0], cells[cur_cells_index][ghost], transferSize * sizeof(float));

		// sequential receive all the data from node 1 to (nodes-1) and place them in the right position
    for (int p = 1; p < nodes; p++) {
      MPI_Recv(final_result[p * num_rows], transferSize, MPI_FLOAT, p, 10, MPI_COMM_WORLD, NULL);
	    }
	// after message receiving, create the final snapshot
	create_snapshot(final_result, real_cols, real_rows, current_iterations);
  }
	// all other nodes
  else {
		// send the main cell(exlude ghost cells) to node 0
    MPI_Send(cells[cur_cells_index][ghost], transferSize, MPI_FLOAT, 0, 10 , MPI_COMM_WORLD);
  }

	// if the iteration hasn't finished go back into the iteration for-loop
	if (current_iterations < (iterations - 1)) goto cont;

	// finalze MPI
	MPI_Finalize();

  // Compute and output the execution time
	time_t end_time = time(NULL);
	printf("\nExecution time: %d seconds\n", (int) difftime(end_time, start_time));

	return 0;
}

// set the hotspot
void hotspot(float **cells, int num_cols, int num_rows, int ghost, int rank, int nodes) {
	// determine which node is the fixed cell in, and the exact position for that cell regarding to a certain node
	if ((hotSpotRow >= (rank * num_rows - ghost)) && (hotSpotRow <= ((rank + 1) * num_rows - ghost))) {
		cells[hotSpotRow - rank * num_rows + ghost][hotSpotCol] = 1000;
	}
}

// Allocates and returns a pointer to a 2D array of floats
float **allocate_cells(int num_cols, int num_rows) {
	float **array = (float **) malloc(num_rows * sizeof(float *));
	if (array == NULL) die("Error allocating array!\n");

	array[0] = (float *) malloc(num_rows * num_cols * sizeof(float));
	if (array[0] == NULL) die("Error allocating array!\n");

	int i;
	for (i = 1; i < num_rows; i++) {
		array[i] = array[0] + (i * num_cols);
	}

	return array;
}

// message passing solution
void send(float **cells, int ghostSize, int ghost, int num_rows, int rank, int nodes) {
	// The message passing tags is designed as below
	// four tags
	// 0 even to odd upward send (lower node to higher node)
	// 1 odd to even upward send
	// 2 even to odd downward send (higher node to lower node)
	// 3 odd to even downward send
	// in this way the message passing can be be finished(optimally) in 4 pass
	// for each tag, all the corresponding send/receive shall happen simultaniously
	// also note that the first/last nodes need to handle differently
	if (rank % 2 == 0) {
		// if it is the first node we only need even to odd upward send and odd to even downward receive
		if (rank == 0) {
			MPI_Send(cells[num_rows], ghostSize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
			MPI_Recv(cells[num_rows + ghost], ghostSize, MPI_FLOAT, 1, 3, MPI_COMM_WORLD, NULL);
		}
		// if it is the last node and it is even
		// we only need odd to even upward receive and even to odd downward send
		else if (rank == nodes - 1) {
			MPI_Recv(cells[0], ghostSize, MPI_FLOAT, nodes - 2, 1, MPI_COMM_WORLD, NULL);
			MPI_Send(cells[ghost], ghostSize, MPI_FLOAT, nodes - 2, 2, MPI_COMM_WORLD);
		}
		else {
			MPI_Send(cells[num_rows], ghostSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
			MPI_Recv(cells[0], ghostSize, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, NULL);
			MPI_Send(cells[ghost], ghostSize, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD);
			MPI_Recv(cells[num_rows + ghost], ghostSize, MPI_FLOAT, rank + 1, 3, MPI_COMM_WORLD, NULL);
		}
	}
	else {
		// if it is the last node and it is odd
		// we only need even to odd upward receive and odd to even downward send
		if (rank == nodes - 1) {
			MPI_Recv(cells[0], ghostSize, MPI_FLOAT, nodes - 2, 0, MPI_COMM_WORLD, NULL);
			MPI_Send(cells[ghost], ghostSize, MPI_FLOAT, nodes - 2, 3, MPI_COMM_WORLD);
		}
		else {
			MPI_Recv(cells[0], ghostSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, NULL);
			MPI_Send(cells[num_rows], ghostSize, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
			MPI_Recv(cells[num_rows + ghost], ghostSize, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD, NULL);
			MPI_Send(cells[ghost], ghostSize, MPI_FLOAT, rank - 1, 3, MPI_COMM_WORLD);
		}
	}
}

// Sets all of the specified cells to their initial value.
// Assumes the existence of a one-cell thick boundary layer.
void initialize_cells(float **cells, int num_cols, int num_rows, int ghost) {
	int x, y;
	for (x = 0; x < num_rows + 2 * ghost; x++) {
		for (y = 0; y <= num_cols + 1; y++) {
			cells[x][y] = INITIAL_CELL_VALUE;
		}
	}
}

// set the fix boundary value (maybe fixed point as well)
void set(float ***cells, int num_cols, int num_rows, int ghost, int rank, int nodes) {
	int x, y, i;
	if (rank == 0) {
		for (x = ghost; x < num_cols + ghost; x++) cells[0][ghost - 1][x] = cells[1][ghost - 1][x] = TOP_BOUNDARY_VALUE;
	}
	if (rank == nodes - 1) {
		for (x = ghost; x < num_cols + ghost; x++) cells[0][num_rows + ghost][x] = cells[1][num_rows + ghost][x] = BOTTOM_BOUNDARY_VALUE;
	}
	for (y = 0; y <= num_rows+1; y++)
		cells[0][y][0] = cells[1][y][0] = LEFT_BOUNDARY_VALUE;
	for (y = 0; y <= num_rows+1; y++)
		cells[0][y][num_cols + 1] = cells[1][y][num_cols + 1] = RIGHT_BOUNDARY_VALUE;
}

// Creates a snapshot of the current state of the cells in PPM format.
// The plate is scaled down so the image is at most 1,000 x 1,000 pixels.
// This function assumes the existence of a boundary layer, which is not
//  included in the snapshot (i.e., it assumes that valid array indices
//  are [1..num_rows][1..num_cols]).
void create_snapshot(float **cells, int num_cols, int num_rows, int id) {
	int scale_x, scale_y;
	scale_x = scale_y = 1;
	// Figure out if we need to scale down the snapshot (to 1,000 x 1,000)
	//  and, if so, how much to scale down
	if (num_cols > 1000) {
		if ((num_cols % 1000) == 0) scale_y = num_cols / 1000;
		else {
			die("Cannot create snapshot for x-dimensions >1,000 that are not multiples of 1,000!\n");
			return;
		}
	}
	if (num_rows > 1000) {
		if ((num_rows % 1000) == 0) scale_x = num_rows / 1000;
		else {
			printf("Cannot create snapshot for y-dimensions >1,000 that are not multiples of 1,000!\n");
			return;
		}
	}

	// Open/create the file
	char text[255];
	sprintf(text, "snapshot.%d.ppm", id);
	FILE *out = fopen(text, "w");
	// Make sure the file was created
	if (out == NULL) {
		printf("Error creating snapshot file!\n");
		return;
	}
	// Write header information to file
	// P3 = RGB values in decimal (P6 = RGB values in binary)
	fprintf(out, "P3 %d %d 100\n", num_cols / scale_x, num_rows / scale_y);
	// Precompute the value needed to scale down the cells
	float inverse_cells_per_pixel = 1.0 / ((float) scale_x * scale_y);
	// Write the values of the cells to the file
	int x, y, i, j;
	for (x = 0; x < num_rows; x += scale_x) {
		for (y = 0; y < num_cols; y += scale_y) {
			float sum = 0.0;
			for (i = x; i < x + scale_x; i++) {
				for (j = y; j < y + scale_y; j++) {
					sum += cells[i][j];
				}
			}
			// Write out the average value of the cells we just visited
			int average = (int) (sum * inverse_cells_per_pixel);
			fprintf(out, "%d 0 %d\t", average, 100 - average);
		}
		fwrite("\n", sizeof(char), 1, out);
	}
	// Close the file
	fclose(out);
}

// directly print the value of the given pointer to the block and its row/column
// print only cell value instead of generate ppm file
// used only for debug process
void printMaps(float **cells, int num_cols, int num_rows, int ranks, int type) {
	char text[255];
	sprintf(text, "maps_in_%d_%d", ranks, type);
	FILE *out = fopen(text, "w");
	// Make sure the file was created
	if (out == NULL) {
		printf("Error creating snapshot file!\n");
		return;
	}
	int x, y;
	for (x = 0; x < num_rows; x++) {
		for (y = 0; y < num_cols; y++) {
			int val = (int)(cells[x][y]);
			fprintf(out, "%d ", val);
		}
		fwrite("\n", sizeof(char), 1, out);
	}
	fclose(out);
}

// Prints the specified error message and then exits
void die(const char *error) {
	printf("%s", error);
	exit(1);
}

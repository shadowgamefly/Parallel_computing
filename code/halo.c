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


// Function prototypes
void print_cells(float **cells, int n_x, int n_y);
void printFinalMaps(float **cells, int n_x, int n_y, int ghost, int iterations);
void printMaps(float **cells, int n_x, int n_y,int rank, int type);
void initialize_cells(float **cells, int n_x, int n_y, int ghost);
void create_snapshot(float **cells, int n_x, int n_y, int id);
void set(float ***cells, int n_x, int n_y, int ghost, int rank, int nodes);
float **allocate_cells(int n_x, int n_y);
void die(const char *error);
void hotspot(float **cells, int n_x, int n_y, int ghost, int rank, int nodes);

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

	// Output the simulation parameters
	// We allocate two arrays: one for the current time step and one for the next time step.
	// At the end of each iteration, we switch the arrays in order to avoid copying.
	// The arrays are allocated with an extra surrounding layer which contains
	// the immutable boundary conditions (this simplifies the logic in the inner loop).
	MPI_Init(&argc, &argv);

	int nodes;
	MPI_Comm_size(MPI_COMM_WORLD, &nodes);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// printf("Grid: %dx%d, Iterations: %d\n running in %d processes\n", real_cols, num_rows, iterations, rank);

	float **cells[2];
	float **final_result;
	int num_cols = real_cols/nodes;
  	int num_rows = real_rows;
	int total_cols = num_cols + 2 * ghost;
	int total_rows = num_rows + 2;
	int transferSize = (num_rows + 2) * num_cols;
	cells[0] = allocate_cells(total_cols, total_rows);
	cells[1] = allocate_cells(total_cols, total_rows);
	if (rank == 0) final_result = allocate_cells(real_cols, real_rows + 2);
	int cur_cells_index = 0, next_cells_index = 1;

	// Initialize the interior (non-boundary) cells to their initial value.
	// Note that we only need to initialize the array for the current time
	//  step, since we will write to the array for the next time step
	//  during the first iteration.
	initialize_cells(cells[0], num_cols, num_rows, ghost);
	// four tags
	// 0 even to odd upward send
	// 1 odd to even upward send
	// 2 even to odd downward send
	// 3 odd to even downward send
	int ghostSize = ghost * (num_rows + 2);
	int transfer = iterations / ghost; // assume always return an int
	int current_iterations = 0;
	
	for (int m = 0; m < transfer ; m++) {
		set(cells, num_cols, num_rows, ghost, rank, nodes);
		if (rank % 2 == 0) {
			if (rank == 0) {
				MPI_Send(cells[cur_cells_index][num_cols], ghostSize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
				MPI_Recv(cells[cur_cells_index][num_cols + ghost], ghostSize, MPI_FLOAT, 1, 3, MPI_COMM_WORLD, NULL);
			}
			else if (rank == nodes - 1) {
				MPI_Recv(cells[cur_cells_index][0], ghostSize, MPI_FLOAT, nodes - 2, 1, MPI_COMM_WORLD, NULL);
				MPI_Send(cells[cur_cells_index][ghost], ghostSize, MPI_FLOAT, nodes - 2, 2, MPI_COMM_WORLD);
			}
			else {
				MPI_Send(cells[cur_cells_index][num_cols], ghostSize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
				MPI_Recv(cells[cur_cells_index][0], ghostSize, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, NULL);
				MPI_Send(cells[cur_cells_index][ghost], ghostSize, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD);
				MPI_Recv(cells[cur_cells_index][num_cols + ghost], ghostSize, MPI_FLOAT, rank + 1, 3, MPI_COMM_WORLD, NULL);
			}
		}
		else {
			if (rank == nodes - 1) {
				MPI_Recv(cells[cur_cells_index][0], ghostSize, MPI_FLOAT, nodes - 2, 0, MPI_COMM_WORLD, NULL);
				MPI_Send(cells[cur_cells_index][ghost], ghostSize, MPI_FLOAT, nodes - 2, 3, MPI_COMM_WORLD);
			}
			else {
				MPI_Recv(cells[cur_cells_index][0], ghostSize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, NULL);
				MPI_Send(cells[cur_cells_index][num_cols], ghostSize, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
				MPI_Recv(cells[cur_cells_index][num_cols + ghost], ghostSize, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD, NULL);
				MPI_Send(cells[cur_cells_index][ghost], ghostSize, MPI_FLOAT, rank - 1, 3, MPI_COMM_WORLD);
			}	
		}
	// Simulate the heat flow for the specified number of iterations
	// printMaps(cells[cur_cells_index], total_cols, total_rows, rank, -m);
		for (i = 0; i < ghost; i++) {
			set(cells, num_cols, num_rows, ghost, rank, nodes);
		// Traverse the plate, computing the new value of each cell
			if ((real_cols > hotSpotCol) && (real_rows > hotSpotRow)) {
				hotspot(cells[cur_cells_index], num_cols, num_rows, ghost, rank, nodes);
			}
			for (x = 1; x < total_cols - 1 ; x++) {
				for (y = 1; y < total_rows - 1; y++) {
					for (iter = 0; iter < iters_per_cell; iter++) {
					//The new value of this cell is the average of the old values of this cell's four neighbors
						cells[next_cells_index][x][y] = (cells[cur_cells_index][x - 1][y]  +
				                 	        	         cells[cur_cells_index][x + 1][y]  +
				                        	        	 cells[cur_cells_index][x][y - 1]  +
				                                 		 cells[cur_cells_index][x][y + 1]) * 0.25;
					}
				}
			}	
		current_iterations += 1;
		if (current_iterations % iterations_per_snapshot == 0) goto createSnap;
		// Swap the two arrays
			cont:
			cur_cells_index = next_cells_index;
			next_cells_index = !cur_cells_index;
	//	printf("Iteration: %d / %d for call %d / %d\n", i + 1, ghost, m + 1, transfer);
		}

	// Print the current progress
	}

  //tag 10 for final transfer, transfer all information to node 0
	createSnap:
  if (rank == 0) {
		memcpy(final_result[0], cells[cur_cells_index][ghost], transferSize * sizeof(float));
    for (int p = 1; p < nodes; p++) {
      MPI_Recv(final_result[p * num_cols], transferSize, MPI_FLOAT, p, 10, MPI_COMM_WORLD, NULL);
    }
	create_snapshot(final_result, real_cols, real_rows, current_iterations);
  }
  else {
    MPI_Send(cells[cur_cells_index][ghost], transferSize, MPI_FLOAT, 0, 10 , MPI_COMM_WORLD);
  }

	if (current_iterations < (iterations - 1)) goto cont;

        create_snapshot(final_result, real_cols, num_rows, 1);
	MPI_Finalize();
	//
	// // Output a snapshot of the final state of the plate
	//
	// // Compute and output the execution time
	time_t end_time = time(NULL);
	printf("\nExecution time: %d seconds\n", (int) difftime(end_time, start_time));

	return 0;
}

void hotspot(float **cells, int num_cols, int num_rows, int ghost, int rank, int nodes) {
	if ((hotSpotCol >= (rank * num_cols - ghost)) && (hotSpotCol <= ((rank + 1) * num_cols - ghost))) {
		cells[hotSpotCol - rank * num_cols + ghost][hotSpotRow] = 1000;
	}
}

// Allocates and returns a pointer to a 2D array of floats
float **allocate_cells(int num_cols, int num_rows) {
	float **array = (float **) malloc(num_cols * sizeof(float *));
	if (array == NULL) die("Error allocating array!\n");

	array[0] = (float *) malloc(num_rows * num_cols * sizeof(float));
	if (array[0] == NULL) die("Error allocating array!\n");

	int i;
	for (i = 1; i < num_cols; i++) {
		array[i] = array[0] + (i * num_rows);
	}

	return array;
}

// Sets all of the specified cells to their initial value.
// Assumes the existence of a one-cell thick boundary layer.
void initialize_cells(float **cells, int num_cols, int num_rows, int ghost) {
	int x, y;
	for (y = 0; y < num_cols + 2 * ghost; y++) {
		for (x = 0; x <= num_rows + 1; x++) {
			cells[y][x] = INITIAL_CELL_VALUE;
		}
	}
}

// set the fix boundary value (maybe fixed point as well)
void set(float ***cells, int num_cols, int num_rows, int ghost, int rank, int nodes) {
	int x, y, i;
	for (x = ghost; x < num_cols + ghost; x++) cells[0][x][0] = cells[1][x][0] = TOP_BOUNDARY_VALUE;
	for (x = ghost; x < num_cols + ghost; x++) cells[0][x][num_rows + 1] = cells[1][x][num_rows + 1] = BOTTOM_BOUNDARY_VALUE;
	if (rank == 0) {
		for (y = 1; y <= num_rows; y++) cells[0][ghost - 1][y] = cells[1][ghost - 1][y] = LEFT_BOUNDARY_VALUE;
	}
	if (rank == nodes - 1) {
		for (y = 1; y <= num_rows; y++) cells[0][num_cols + ghost][y] = cells[1][num_cols + ghost][y] = RIGHT_BOUNDARY_VALUE;
	}


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
		if ((num_cols % 1000) == 0) scale_x = num_cols / 1000;
		else {
			die("Cannot create snapshot for x-dimensions >1,000 that are not multiples of 1,000!\n");
			return;
		}
	}
	if (num_rows > 1000) {
		if ((num_rows % 1000) == 0) scale_y = num_rows / 1000;
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
	for (y = 0; y < num_rows; y += scale_y) {
		for (x = 0; x < num_cols; x += scale_x) {
			float sum = 0.0;
			for (j = y; j < y + scale_y; j++) {
				for (i = x; i < x + scale_x; i++) {
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

void printFinalMaps(float **cells, int num_cols, int num_rows, int ghost, int iterations) {
	char text[255];
	sprintf(text, "Final_map_%d_%d", ghost, iterations);
	FILE *out = fopen(text, "w");
	// Make sure the file was created
	if (out == NULL) {
		printf("Error creating snapshot file!\n");
		return;
	}
	int x, y;
	for (x = 1; x < num_rows - 1; x++) {
		for (y = 0; y < num_cols; y++) {
			int val = (int)(cells[y][x]);
			fprintf(out, "%d ", val);
		}
		fwrite("\n", sizeof(char), 1, out);
	}
	fclose(out);
}

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
			int val = (int)(cells[y][x]);
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

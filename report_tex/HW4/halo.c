/* initial author: Prof. Andrew Grimshaw (ag8t)
  modified by Jerry Sun (ys7va)
  2017.05.04
  halo.c takes 4 inputs as num_thread, dim_x, dim_y and iterations and it will
  perform a iterations number of heat simulation on a plate of dim_x * dim_y size using num_thread threads
  It will print the final runtime for this program.
  This program has two versions with different memory allocation strategies, detailed can be found
  within **allocate_cells & **allocate_cells_numa
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <numa.h>
#include "pthread_bar.c"

/* predefined condition for HALO */
#define TOP_BOUNDARY 0
#define BOTTOM_BOUNDARY 100
#define LEFT_BOUNDARY 0
#define RIGHT_BOUNDARY 100
#define INITIAL_VALUE 50
#define HOTSPOT_ROW 4500
#define HOTSPOT_COL 6500
#define HOTSPOT_TEMP 1000

/*method header declaration, detail explanation can be found after the main method*/

/*exit method for possible errors*/
void die(const char *error);

/*helper to create ppm file for visualization*/
void create_snapshot(float **cells, int num_cols, int num_rows, int id);

/*set predefined conditions on a given cell*/
void set(float ***cells, int num_rows, int num_cols);

/*naive memory allocation based on the sequential version provided*/
float **allocate_cells_naive(int num_cols, int num_rows);

/*optimized memory allocation using numa_alloc_onnode*/
float **allocate_cells_numa(int num_cols, int num_rows, int *sep);

/*main computing funtion for each thread*/
static void *thread_compute(void *arg);

/*global variable declaration*/

int real_row, real_col;        /*The real size of the plate we are going to simulate*/
int total_row, total_col;      /*The actual size of the plate we are going to simulate, including fixed border*/
int iterations;                /*Total number of iterations*/
int num_threads;               /*Total number of threads*/
int hotspot_id = -1;           /*The thread_id of a thread to handle hotspot, -1 if not applicable */
pthread_barrier_t barrier;     /*global barrier to synchronize the threads for each iterations*/

/*the struct to pass into each thread*/
typedef struct{
  int start_row, end_row;      /*The rows this thread handles inclusive for start, exclusive for end*/
  int thread_id;               /*The id of this thread*/
  float ***cell;               /*Two cells used for simulation*/
} thread_info;


int main(int argc, char **argv) {
  /*Read the comman line argument into global variable, if not satisfied, exit.*/
  if (argc != 5) die("argument incomplete");
  num_threads = atoi(argv[1]);
  real_row = atoi(argv[2]);
  real_col = atoi(argv[3]);
  iterations = atoi(argv[4]);

  int i, x, y;                  /*initialize some counters used later*/
  float **cell[2];              /*Two main cells to store simulation values*/
  int sep[num_threads + 1];     /*list to set the border (in row major order) of each thread*/
  total_row = real_row + 2;     /*Set the actual plate row*/
  total_col = real_col + 2;     /*Set the actual plate col*/

  /*Compute the separation boundary in rows and stored in a list
    We want to evenly distribute the task into threads
    sep[i] specify the row to start for ith thread
    sep[num_threads] marks the end of the last thread */
  int start_row = 1;
  int increment = real_row/num_threads;
  for (i = 0; i < num_threads; i++) {
    sep[i] = start_row;
    start_row = start_row + increment;
  }
  sep[num_threads] = real_row + 1;

  pthread_t threads[num_threads]; /*initialize specifc number of threads asked*/
  thread_info data[num_threads];  /*initialize corresponding number of thread_info to pass into thread*/
  pthread_attr_t attrs;           /*misc for thread initialization*/
  pthread_attr_init(&attrs);      /*misc for thread initialization*/
  void *status; /*misc for thread join*/
  pthread_barrier_init(&barrier, NULL, num_threads); /*initialize the barrier with corresponding number of threads*/

  /*Start the timer*/
  time_t start_time = time(NULL);

  /*Allocate two cell blocks
    In each cell block, the distribution of rows of memory is defined by list sep
    So that different memory partitions are actually on different nodes
    corresponding to later threads computation */
  cell[0] = allocate_cells_numa(total_col, total_row, sep);
  cell[1] = allocate_cells_numa(total_col, total_row, sep);

  /*set the precondition for both cells*/
  set(cell, real_row, real_col);

  /*loop through all the threads_id, initialize the thread info, and then initialize all the corresponding threads*/
	for (i = 0; i < num_threads; i++) {
    /* initialize the info that is going to be passed into threads */
    data[i].cell = cell;
    data[i].thread_id= i;
    data[i].start_row = sep[i];
    data[i].end_row = sep[i + 1];

    /*determine the thread that is going to hand hotspot
      if not applicable then hotspot_id = -1, as default */
    if (HOTSPOT_ROW > data[i].start_row && HOTSPOT_ROW < data[i].end_row) hotspot_id = i;

    /*initialize thread within the corresponding info, if fail then quit*/
    int response = pthread_create(&threads[i], &attrs, thread_compute, (void*) &data[i]);
    if (response) die("thread initialization fail\n");
  }

  /*join all the threads initiated*/
  for (i = 0; i < num_threads; i++) {
    int response = pthread_join(threads[i], &status);
    if (response) die("error when joining a thread\n");
  }

  /*stop the timer*/
  time_t end_time = time(NULL);

  /*print the total runtime, and create the snapshot to check the final result*/
  printf("\nExecution time: %d seconds\n", (int) difftime(end_time, start_time));
  create_snapshot(cell[iterations % 2 == 0 ? 0 : 1], real_row, real_col, iterations);
}


/* main computing funtion for each thread */
static void *thread_compute(void *info) {
  int i, j, m;  /*general loop counter*/

  /*cast the passing info*/
  thread_info *data = (thread_info*) info;

  /*This line is only needed when using numa
  memory allocation */

  /*bind the thread onto a particular node
    hermes has 8 nodes, each has 8 cores,
    so we deploy 8 thread onto a single node
    by thread_id:
    0, 8, 16 ... goto node 0
    1, 9, 17 ... goto node 1 ... */
  int resp = numa_run_on_node((data->thread_id) % 8);
  /*Note: This line is only needed when using numa
  memory allocation */


  /*initital the index for two cell boxes
    for each iteration, the value will be computed
    based on the cur_cell, and saved into the next_cell
    and at the end of a single iteration, the cur and next index
    will flip, and continue */
  int cur_cells_index = 0;
  int next_cells_index = 1;
  float ***cell = data -> cell;

  /*main iterations*/
  for (m = 0; m < iterations; m++) {
    /*if it is the hotspot thread, it needs to set the HOTSPOT*/
    if (hotspot_id == data -> thread_id) {
      cell[cur_cells_index][HOTSPOT_ROW][HOTSPOT_COL] = HOTSPOT_TEMP;
    }

    /*each thread only need to work on the corresponding rows they are assigned*/
    for (i = data->start_row; i < data->end_row; i++) {
      for (j = 1; j < total_col - 1; j++) {
        cell[next_cells_index][i][j] = (cell[cur_cells_index][i][j-1]
          + cell[cur_cells_index][i][j+1]
          + cell[cur_cells_index][i-1][j]
          + cell[cur_cells_index][i+1][j]) * 0.25;
      }
    }

    /*indx flip*/
    cur_cells_index = next_cells_index;
    next_cells_index = !cur_cells_index;

    /*barrier for thread synchronization, for each iteration, */
    int s = pthread_barrier_wait(&barrier);
  }
  pthread_exit(NULL);
}

/* prints an error message and exits the program */
void die(const char *error) {
  printf("%s", error);
  exit(1);
}

/* Creates a snapshot of the current state of the cells in PPM format.
 The plate is scaled down so the image is at most 1,000 x 1,000 pixels.
 This function assumes the existence of a boundary layer, which is not
  included in the snapshot (i.e., it assumes that valid array indices
  are [1..num_rows][1..num_cols]).*/
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

/*set the initial value to both cells, except the hotspot
  hotspot is assigned at thread runtime*/
void set(float ***cells, int num_rows, int num_cols) {
  int x, y, i;
  /* Boundary value */
  for (x = 1; x <= num_cols; x++) cells[0][0][x] = cells[1][0][x] = TOP_BOUNDARY;
  for (x = 1; x <= num_cols; x++) cells[0][num_rows + 1][x] = cells[1][num_rows + 1][x] = BOTTOM_BOUNDARY;
  for (y = 1; y <= num_rows; y++) cells[0][y][0] = cells[1][y][0] = LEFT_BOUNDARY;
  for (y = 1; y <= num_rows; y++) cells[0][y][num_cols + 1] = cells[1][y][num_cols + 1] = RIGHT_BOUNDARY;

  /* Internal value */
  for (x = 1; x <= num_rows; x++)
		for (y = 1; y <= num_cols; y++)
			cells[0][x][y] = cells[1][x][y] = INITIAL_VALUE;
}

/* Allocates and returns a pointer to a 2D array of floats, original version*/
float **allocate_cells_naive(int num_cols, int num_rows) {
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

/* Allocates and returns a pointer to a 2D array of floats using numa*/
float **allocate_cells_numa(int num_cols, int num_rows, int *sep) {
  /*some general counter*/
  int counter = 0;
  int i;

  /*number of floats need to be allocated*/
  int to_alloc = 0;

  /*node the memory is going to be allocated, all nodes are allocated on node i % 8 */
  int node = 0;
  /*First allocate all row pointers */
  float **array = (float **) malloc(num_rows * sizeof(float *));
  if (array == NULL) die("Error allocating array!\n");

  /*allocate the first row*/
  int to_alloc = (sep[1] - sep[0] + 1) * num_cols;
  array[0] = (float *) numa_alloc_onnode(to_alloc * sizeof(float), node);
  if (array[0] == NULL) die("Error allocating array!\n");
  for (counter = 1; counter < sep[1]; counter++) {
    array[counter] = array[0] + (counter * num_cols);
  }

  /*allocate the second to the second last row*/
  for (i = 1; i < num_threads - 1; i++) {
    int increment = sep[i + 1] - sep[i];
    to_alloc = increment * num_cols;
    node = i % 8;
    array[sep[i]] = (float *) numa_alloc_onnode(to_alloc * sizeof(float), node);
    if (array[sep[i]] == NULL) die("Error allocating array!\n");
    /*assign value pointers to row pointers*/
    for (counter = sep[i] + 1; counter < sep[i + 1]; counter++) {
      array[counter] = array[sep[i]] + (counter - sep[i]) * num_cols;
    }
  }

  /*allocate the last row*/
  to_alloc = (sep[num_threads] - sep[num_threads - 1] + 1) * num_cols;
  node = (num_threads - 1) % 8
  array[sep[num_threads - 1]] = (float *) numa_alloc_onnode(to_alloc * sizeof(float), node);
  /*assign value pointers to row pointers*/
  for (counter = sep[num_threads - 1] + 1; counter <= sep[num_threads]; counter++) {
    array[counter] = array[sep[num_threads - 1]] + (counter - sep[num_threads - 1]) * num_cols;
  }

  return array;
}

#include <stdio.h>
#include <mpi.h>

//multiply n x n matrices
//John Lim
//Parallel Lab

#define N 500 //Number of rows and columns for matrices a and b.

#define MASTER_TO_SLAVE_TAG 1
#define SLAVE_TO_MASTER_TAG 4

void make_matrices(); //makes matrices a and b.
void printArray(); //prints the resulting matrix, c.

int rank; //process rank
int size; //number of processes
int i, j, k;

double a[N][N];
double b[N][N];
double c[N][N]; //result of multiplication of a and b.

double start_time;
double end_time;

int low_bound; //lower bound of # rows of matrix a to a  slave
int upper_bound; //upper bound of # of rows of matrix a to a slave 
int portion; //portion of # of rows of matrix a to a slave

MPI_Status status;
MPI_Request request;

int main(int argc, char *argv[]) {
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		//Master Code 
		make_matrices();
		start_time = MPI_Wtime();
		for (i = 1; i < size; i++) {
			portion = (N/(size - 1));
			low_bound = (i - 1) * portion;
			if (((i+1) == size) && ((N % (size-1)) != 0)) {
				upper_bound = N;
			} else {
				upper_bound = low_bound + portion;
			}

			MPI_Isend(&low_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &request);
			MPI_Isend(&upper_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG + 1, MPI_COMM_WORLD, &request);
			MPI_Isend(&a[low_bound][0], (upper_bound - low_bound) * N, MPI_DOUBLE, i, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &request);
		}
	}

	MPI_Bcast(&b, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	

	if (rank > 0) {
		//Slave Code
		MPI_Recv(&low_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&upper_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG + 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&a[low_bound][0], (upper_bound - low_bound) * N, MPI_DOUBLE, 0, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &status);
		for (i = low_bound; i < upper_bound; i++) {
			for(j = 0; j < N; j++) {
				for (k = 0; k < N; k++) {
					c[i][j] += (a[i][k] * b[k][j]);
				}
			}
		}

		MPI_Isend(&low_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &request);
		MPI_Isend(&upper_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG + 1, MPI_COMM_WORLD, &request);
		MPI_Isend(&c[low_bound][0], (upper_bound - low_bound) * N, MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &request);
	}

	if (rank == 0) {
		//Master Code finalizing
		for (i = 1; i < size; i++) {
			
			MPI_Recv(&low_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(&upper_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG + 1, MPI_COMM_WORLD, &status);
			MPI_Recv(&c[low_bound][0], (upper_bound - low_bound) * N, MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &status);
		}
		
		end_time = MPI_Wtime();
		printf("Total Running Time: %f\n", (end_time - start_time));
//		printArray();
	}

	MPI_Finalize();
	return 0;
}

void make_matrices() {
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a[i][j] = i + j;
		}	
	}
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			b[i][j] = i * j;
		}
	}
}

void printArray() {
	for (i = 0; i < N; i++) {
		printf("\n");
		for (j=0; j < N; j++) {
			printf("%8.2f  ", a[i][j]);
		}
	}
	printf("\n\n");
	for (i = 0; i < N; i++) {
		printf("\n");
		for (j = 0; j < N; j++) {
			printf("%8.2f  ", b[i][j]);
		}
	}
	printf("\n\n");
	for (i = 0; i < N; i++) {
		printf("\n");
		for (j = 0; j < N; j++) {
			printf("%8.2f  ", c[i][j]);
		}
	}
	printf("\n\n");
}

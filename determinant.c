#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

double Partition(double **a, int s, int end, int n);

int main(int argc, char* argv[])
{
    int NumberProcesses,    /* number of tasks in partition */
            rank,        /* a task identifier */
            NumberWorkers,    /* number of worker tasks */
            destination,        /* task id of message destinationination */
            offset,
            i, j, k;    /* misc */
    double StartTime, EndTime, read_StartTime, read_EndTime, print_StartTime, print_EndTime;
    double det, **matrix, *buffer, determinant_of_matrix;
    int n;    /*number of rows and columns in matrix */
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumberProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    StartTime = MPI_Wtime();
    NumberWorkers = NumberProcesses - 1;

    if (!rank)//rank =0 
    {
        read_StartTime = MPI_Wtime();
        FILE *fp=fopen("mpimatrix.txt","r");
        fscanf(fp, "%d", &n);
        buffer = (double *) malloc(sizeof(double) * n * n);
        for(i=0;i<n;i++)
        	for(j=0;j<n;j++)
        		fscanf(fp, "%lf",  buffer+n*i+j);
        read_EndTime = MPI_Wtime();
        printf("Number of  tasks = %d\n", NumberProcesses);
        print_StartTime = MPI_Wtime();
        printf("Entered matrix\n");
        for (i = 0; i < n; i++) 
        {
            for (j = 0; j < n; j++)
            	printf("%lf\t",buffer[i*n+j]);
            printf("\n");
        }
        print_EndTime = MPI_Wtime();

        /* send matrix data to the worker tasks */
        float temp;
        temp = n / NumberProcesses;
        temp += 0.5;
        offset = temp; //offset is int

        for (destination = 1; destination <= NumberWorkers; destination++) 
        {
            MPI_Send(&n, 1, MPI_INT, destination, 1, MPI_COMM_WORLD);
            MPI_Send(buffer, n * n, MPI_DOUBLE, destination, 1, MPI_COMM_WORLD);
        }

        //allocation of memory
        matrix = (double **) malloc((n) * sizeof(double[n]));
        for (k = 0; k < n; ++k)
            matrix[k] = (double *) malloc((n) * sizeof(double));

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                matrix[i][j] = buffer[i * n + j];
        determinant_of_matrix = Partition(matrix, 0, offset, n);

        free(buffer);

        for (i = 1; i <= NumberWorkers; i++) 
        {
            MPI_Recv(&det, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
            determinant_of_matrix += det;
        }

        EndTime = MPI_Wtime();
        printf("Determinant of matrix is : %3.2lf\n", determinant_of_matrix);
        printf("Elapsed time is %f\n",
               ((EndTime - StartTime) - (print_EndTime - print_StartTime) - (read_EndTime - read_StartTime)));

    }
    if (rank) 
    {
        MPI_Recv(&n, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        buffer = (double *) malloc(sizeof(double) * n * n);
        MPI_Recv(buffer, n * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        float temp;
        temp = n / NumberProcesses;
        temp += 0.5;
        offset = temp;

        int end;
        int start = (rank) * offset;
        if ((rank) == NumberWorkers)
            end = n;
        else
            end = (start + offset);

        matrix = (double **) malloc((n) * sizeof(double[n]));
        for (k = 0; k < n; ++k)
            matrix[k] = (double *) malloc((n) * sizeof(double));

        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++) 
                matrix[i][j] = buffer[i * n + j];


        det = Partition(matrix, start, end, n);
        int h = 0;
        for (h = 0; h < n; ++h)
            free(matrix[h]);
        free(matrix);
        free(buffer);
        MPI_Send(&det, 1, MPI_DOUBLE,0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}


double MatrixDeterminant(int nDim, double *pfMatr) 
{
    double fDet = 1.;
    double fMaxElem;
    double fAcc;
    int i, j, k, m;

    for (k = 0; k < (nDim - 1); k++) // base row of matrix
    {
        // search of row with max element
        fMaxElem = fabs(pfMatr[k * nDim + k]);
        m = k;
        for (i = k + 1; i < nDim; i++) 
        {
            if (fMaxElem < fabs(pfMatr[i * nDim + k])) 
            {
                fMaxElem = pfMatr[i * nDim + k];
                m = i;
            }
        }

        if (m != k)  
        {
            for (i = k; i < nDim; i++) 
            {
                fAcc = pfMatr[k * nDim + i];
                pfMatr[k * nDim + i] = pfMatr[m * nDim + i];
                pfMatr[m * nDim + i] = fAcc;
            }
            fDet *= (-1.);
        }

        if (pfMatr[k * nDim + k] == 0.) 
        	return 0.0;

        // trianglulation of matrix
        for (j = (k + 1); j < nDim; j++) // current row of matrix
        {
            fAcc = -pfMatr[j * nDim + k] / pfMatr[k * nDim + k];
            for (i = k; i < nDim; i++) 
            {
                pfMatr[j * nDim + i] = pfMatr[j * nDim + i] + fAcc * pfMatr[k * nDim + i];
            }
        }
    }
    for (i = 0; i < nDim; i++)
        fDet *= pfMatr[i * nDim + i]; // diagonal elements multiplication
    return fDet;
}
				//(matrix    ,start ,offset   ,n)
double Partition(double **a, int s, int end, int n) 
{
    int i, j, j1, j2;
    double det = 0;
    double **m = NULL;

    det = 0;                      // initialize determinant of sub-matrix
    // for each column in sub-matrix
    for (j1 = s; j1 < end; j1++) 
    {
        // get space for the pointer list
        m = (double **) malloc((n - 1) * sizeof(double *));
        for (i = 0; i < n - 1; i++)
            m[i] = (double *) malloc((n - 1) * sizeof(double));
        for (i = 1; i < n; i++) //removes that row and column; i=1 as first row never selected
        {
            j2 = 0;
            for (j = 0; j < n; j++) 
            {
                if (j == j1) 
                	continue;
                m[i - 1][j2] = a[i][j];
                j2++;
            }
        }
        int dim = n - 1;
        double fMatr[dim * dim];
        for (i = 0; i < dim; i++)
            for (j = 0; j < dim; j++)
                fMatr[i * dim + j] = m[i][j];
        det += pow(-1.0, 1.0 + j1 + 1.0) * a[0][j1] * MatrixDeterminant(dim, fMatr);
        for (i = 0; i < n - 1; i++) 
        	free(m[i]);
        free(m);
    }
    return (det);
}

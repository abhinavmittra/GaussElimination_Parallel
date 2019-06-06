Gauss Elimination By: Abhinav Mittra
MPI: Calculates the determinant using Gauss Elimination method
CUDA: Solves n equations of n variables in parallel

Steps to execute MPI:

1.) Generate a random matrix of order n by running the randmpi.c
2.) Run the determinant.c program using "mpicc determinant.c -lm" to compile the program.

Steps to execute CUDA:

1.) Generate a random matrix of order n by running the randcuda.c
2.) Run the gauss.cu program using "nvcc gauss.cu filename" to compile the program.

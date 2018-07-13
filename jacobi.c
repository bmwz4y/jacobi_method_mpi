#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#define MAXSIZE 10 				// max size of dimension for matrix A
#define MAXDELTA 0.001 			// error margin
#define MAXITERATIONS 1000000 	// always use large number (except for debugging)
#define TAG 101 				// tag for MPI_Send
#define TIMESTOCOMMUNICATE 10 	// how many times processors will communicate while trying to find answer

// function prototypes
float jacobi_sum(float matrixA[][MAXSIZE], float oldX[], int i, int order);
float jacobi_delta(float newX[], float oldX[], int order);

void main(int argc, char *argv[])
{
	/* jacobi method solving a system of linear equations */
	// A must be strongly diagonal dominant
	// 3 6 -2 1 -2 7 2 1 2 -5 11 5 -1 <== test string answer should be 2 1 1
	// 2 3 1 1 2 -1.25 0 <== test string answer should be -0.5 0.25

	// use the following with MAXITERATIONS 2 and -np 8 to see that guess made by proc 2 increased speed of convergence
	// 2 2 1 0 2 8 4 <== test string answer should be 3 2

	int myid, numprocs, order, iteration, i, j, myWave;
	float matrixA[MAXSIZE][MAXSIZE], vectorB[MAXSIZE], tempX1[MAXSIZE], tempX2[MAXSIZE], myInitGuess;
	float *oldX, *newX, *tempFloatPointer;//, *gatherBuff;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	if (myid == 0)
	{
		printf("Enter order of system [integer up to %d]:\n", MAXSIZE);
		scanf("%d", &order);

		// to prevent error// not working
		/*
		// to prevent error and shows feedback message
		if (order > MAXSIZE || order < 1)
		{
			printf("Order of system must be between 1 and %d\nTry again\n", MAXSIZE);
		}
		*/
	}

	// to prevent error// not working
	/*
	if (order > MAXSIZE || order < 1)
	{
		MPI_Finalize();
		return;
	}
	*/

	MPI_Bcast(&order, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// uses only portion of order from matrixA
	// easier and less error than dynamic memory allocation
	if (myid == 0)
	{
		printf("Enter the matrix A (Left-hand side):\n");
		for (i = 0; i < order; i++)
			for (j = 0; j < order; j++)
				scanf("%f", &matrixA[i][j]);

		printf("Enter the vector b (Right-hand side):\n");
		for (i = 0; i < order; i++)
			scanf("%f", &vectorB[i]);
	}

	MPI_Bcast(&matrixA, order*MAXSIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Bcast(&vectorB, order, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// idea 1:
	// initialize first iteration x vector with special data in respect of myid instead of always zero
	// the idea is making initial guess span area around potential convergence with the increasing number of processors
	if (myid == 0)
	{
		float maxNumInA = matrixA[0][0];
		float maxNumInB = vectorB[0];

		for (i = 0; i < order; i++)
		{
			for (j = 0; j < order; j++)
				if (matrixA[i][j] > maxNumInA)
					maxNumInA = matrixA[i][j];
			if (vectorB[i] > maxNumInB)
				maxNumInB = vectorB[i];
		}
		if (maxNumInA > maxNumInB)
			myInitGuess = maxNumInA;
		else
			myInitGuess = maxNumInB;
	}
	MPI_Bcast(&myInitGuess, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	myInitGuess *= (float)myid / (float)numprocs;
	if (myid % 2 == 1)
		myWave *= -1;
	else
		myWave = 1;
	myInitGuess *= myWave;
	// to debug uncomment the following line
	//printf("@ %d myInitGuess = %f\n", myid, myInitGuess);
	for (i = 0; i < order; i++)
		tempX1[i] = myInitGuess;
	// end idea 1

	// first swap because of do while loop
	newX = tempX1;
	oldX = tempX2;

	iteration = 0;
	do
	{
		iteration++;

		// swap oldX with newX
		tempFloatPointer = oldX;
		oldX = newX;
		newX = tempFloatPointer;

		for (i = 0; i < order; i++)
		{
			newX[i] = (vectorB[i] - jacobi_sum(matrixA, oldX, i, order)) / matrixA[i][i];

			// to debug uncomment the following lines
			//printf("@iteration(%d), processor(%d), oldX[%d] (%f)\n", iteration, myid, i, oldX[i]);
			//printf("@iteration(%d), processor(%d), newX[%d] (%f)\n", iteration, myid, i, newX[i]);
		}

		// idea 2:
		// every percentage of total runtime communicate processors to share knowledge and not last run and more than one proccessor
		if (iteration % (MAXITERATIONS/TIMESTOCOMMUNICATE) == 0 && iteration != MAXITERATIONS && numprocs > 1)
		{
			// pseudo code: (sig fault error)
			// the proccessor with least delta should send to all other 
			// while when they recieve make them change it to + and - percentage based on their rank in repect to total numprocs
			/*
			float secondGuessOffset;
			float myDelta = jacobi_delta(newX, oldX, order);
			int rankWithLeastDelta = 0;
			// to debug uncomment the following line
			printf("@%d   before malloc\n", myid);
			if (myid == 0)
				gatherBuff = malloc(sizeof(float) * numprocs);
			// to debug uncomment the following line
			printf("@%d   before gather\n", myid);
			MPI_Gather(&myDelta, 1, MPI_FLOAT, &gatherBuff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
			printf("myDelta%f\n", myDelta);
			printf("gatherBuff0%f\n", gatherBuff[0]);
			printf("gatherBuff numprocs%f\n", gatherBuff[numprocs-1]);
			// to debug uncomment the following line
			printf("@%d   after gather\n", myid);
			if (myid == 0)
			{
				float leastDelta = gatherBuff[0];
				for (i = 1; i < numprocs; i++)
					if (gatherBuff[i] < leastDelta)
					{
						leastDelta = gatherBuff[i];
						rankWithLeastDelta = i;
					}
			}

			MPI_Bcast(&rankWithLeastDelta, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&newX, order, MPI_FLOAT, rankWithLeastDelta, MPI_COMM_WORLD);

			for (i = 0; i < order; i++)
			{
				secondGuessOffset = newX[i];
				secondGuessOffset += myWave * (newX[i] * (float)myid / (float)numprocs);
			}
			*/
			// to debug uncomment the following line
			//printf("loop %d for id %d \n", iteration, myid);
		}
		//end idea 2
	} while ((iteration < MAXITERATIONS) && (jacobi_delta(newX, oldX, order) > MAXDELTA));

	if (jacobi_delta(newX, oldX, order) <= MAXDELTA)
	{
		MPI_Send(newX, order, MPI_FLOAT, 0, TAG, MPI_COMM_WORLD);
		// to debug uncomment the following lines
		//for (i = 0; i < order; i++)
		//	printf("@processor(%d), has found answer with newX[%d] = %f\n", myid, i, newX[i]);
	}

	// to make sure all processors are done before printing
	MPI_Barrier(MPI_COMM_WORLD);

	if (myid == 0)
	{
		MPI_Request myRequest;
		int myRequestHasRecieved = 0;// flag
		float *vectorX = NULL;

		MPI_Irecv(&tempX1, order, MPI_FLOAT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &myRequest);
		MPI_Request_get_status(myRequest, &myRequestHasRecieved, MPI_STATUS_IGNORE);
		if (myRequestHasRecieved)
			vectorX = tempX1;
		// to debug uncomment the following lines
		//for (i = 0; i < order; i++)
		//{
		//	printf("@processor(%d), has tempX1[%d] = %f\n", myid, i, tempX1[i]);
		//	printf("@processor(%d), while vectorX[%d] = %f\n", myid, i, vectorX[i]);
		//}

		// if pointer is not null then answer was found print it else feedback message
		if (vectorX != NULL)
		{
			printf("The solution for vector x is:\n");
			for (i = 0; i < order; i++)
				printf("x%d = %3.3f\n", i+1, vectorX[i]);
			printf("\n");
		}
		else
			printf("Did %d iterations with %1.3f error margin, but no answer!\n", MAXITERATIONS, MAXDELTA);
	}
	
	//free(gatherBuff);
	MPI_Finalize();
}

float jacobi_sum(float matrixA[][MAXSIZE], float oldX[], int i, int order)
{
	int j;
	float sum = 0.0;

	for (j = 0; j < order; j++)
		if (i != j)
			sum += matrixA[i][j] * oldX[j];
	return sum;
}

float jacobi_delta(float newX[], float oldX[], int order)
{
	int i;
	float maxDelta = 0.0, delta;

	for (i = 0; i < order; i++)
	{
		delta = newX[i] - oldX[i];

		// delta must be absolute
		if (delta < 0)
			delta *= -1;

		// largest delta between individual x entries is delta for all
		if (delta > maxDelta)
			maxDelta = delta;
	}

	return maxDelta;
}

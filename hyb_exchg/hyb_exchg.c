#include "hyb_exchg.h"
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

/*
 * Initialisation/destruction d'une structure shared_exchg_t
 * nthreads : nombre de threads (du processus MPI) qui vont participer a l'echange
 */
void shared_exchg_init(shared_exchg_t *sh_ex, int nthreads)
{
  // Init buffer value to store left and right
  sh_ex->left = 0.0;
  sh_ex->right = 0.0;

  // Init the number of threads
  sh_ex->nthreads = nthreads;

  // Just wait after second sem_wait
  sh_ex->sem = malloc(sizeof(sem_t));
  sem_init(sh_ex->sem, 1, 1);
  sh_ex->first = 1;
  sh_ex->terminate = 0;
}

void shared_exchg_destroy(shared_exchg_t *sh_ex)
{
  // Destroy the semaphore
  sem_destroy(sh_ex->sem);
  free(sh_ex->sem);
}


/*
 * Echange hybride MPI/pthread
 * Si processus MPI existe "a gauche", lui envoie la valeur sh_arr[0] et recoit de lui *val_to_rcv_left
 * Si processus MPI existe "a droite", lui envoie la valeur sh_arr[mpi_decomp->mpi_nloc-1] et recoit de lui *val_to_rcv_right
 * Si processus voisin n'existe pas, valeur correspondante affectee a 0
 */
void hyb_exchg(
               double *sh_arr,
               shared_exchg_t *sh_ex,
               double *val_to_rcv_left, double *val_to_rcv_right,
               mpi_decomp_t *mpi_decomp)
{
  /**************************/
  /* Master thread election */
  /**************************/

  sem_wait(sh_ex->sem);

  // You are the first thread, you will exchange with other first thread of mpi process
  if (sh_ex->first)
    {
      sh_ex->first = 0;

      /****************/
      /* MPI Exchange */
      /****************/

      int head = 0;
      int tail = mpi_decomp->mpi_nproc - 1;

      if (head == tail) /* 1 process */
        {
          // Nothing to receive 
          *val_to_rcv_left = 0.0;
          *val_to_rcv_right = 0.0;
        }
      else if (mpi_decomp->mpi_rank == head) /* Head process */
        {
          // Nothing to receive from left
          *val_to_rcv_left = 0.0;
          
          // Send to right
          MPI_Send(&(sh_arr[mpi_decomp->mpi_nloc - 1]), 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
          
          // Recv from right
          MPI_Recv(&(sh_ex->right), 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      else if (mpi_decomp->mpi_rank == tail) /* Tail process */
        {
          // Nothing to receive from right
          *val_to_rcv_right = 0.0;
          
          // Recv from left
          MPI_Recv(&(sh_ex->left), 1, MPI_DOUBLE, mpi_decomp->mpi_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          // Send to left
          MPI_Send(&(sh_arr[0]), 1, MPI_DOUBLE, mpi_decomp->mpi_rank - 1, 0, MPI_COMM_WORLD);
        }
      else /* Middle process */
        {
          // Recv from left
          MPI_Recv(&(sh_ex->left), 1, MPI_DOUBLE, mpi_decomp->mpi_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          // Send to right
          MPI_Send(&(sh_arr[mpi_decomp->mpi_nloc - 1]), 1, MPI_DOUBLE, mpi_decomp->mpi_rank + 1, 0, MPI_COMM_WORLD);

          // Recv from right
          MPI_Recv(&(sh_ex->right), 1, MPI_DOUBLE, mpi_decomp->mpi_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          
          // Send to left
          MPI_Send(&(sh_arr[0]), 1, MPI_DOUBLE, mpi_decomp->mpi_rank - 1, 0, MPI_COMM_WORLD);
        }

      /****************/
      /* Update value */
      /****************/
      
      *val_to_rcv_left = sh_ex->left;
      *val_to_rcv_right = sh_ex->right;

      // Release next thread
      sh_ex->terminate++;
      sem_post(sh_ex->sem);
    }
  else // Not the first thread
    {
      /****************/
      /* Update value */
      /****************/
      
      *val_to_rcv_left = sh_ex->left;
      *val_to_rcv_right = sh_ex->right;

      // Check if we are the last thread and re-init variable
      sh_ex->terminate++;

      if (sh_ex->terminate == sh_ex->nthreads)
        {
          sh_ex->first = 1;
          sh_ex->terminate = 0;
        }
      
      // Release next thread
      sem_post(sh_ex->sem);      
    }
}

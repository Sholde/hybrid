#include "hyb_reduc.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void shared_reduc_init(shared_reduc_t *sh_red, int nthreads, int nvals)
{
  // Init variable
  sh_red->nvals = nvals;
  sh_red->nthreads = nthreads;

  // Init shared array
  sh_red->red_val = malloc(sizeof(double) * nvals);
  memset(sh_red->red_val, 0, sizeof(double) * nvals);

  // Init mutex
  sh_red->red_mut = malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(sh_red->red_mut, NULL);

  // Init barrier
  sh_red->red_bar = malloc(sizeof(pthread_barrier_t));
  pthread_barrier_init(sh_red->red_bar, NULL, nthreads);

  // Init semaphore
  sh_red->sem = malloc(sizeof(sem_t));
  sem_init(sh_red->sem, 0, 1);
  sh_red->terminate = 0;
  sh_red->set_master = 0;
}

void shared_reduc_destroy(shared_reduc_t *sh_red)
{
  // Destroy semaphore
  sem_destroy(sh_red->sem);
  free(sh_red->sem);

  // Destroy barrier
  pthread_barrier_destroy(sh_red->red_bar);
  free(sh_red->red_bar);

  // Destroy mutex
  pthread_mutex_destroy(sh_red->red_mut);
  free(sh_red->red_mut);

  // Release shared array
  free(sh_red->red_val);
}

/*
 * Reduction  hybride MPI/pthread
 * in  : tableau des valeurs a reduire (de dimension sh_red->nvals)
 * out : tableau des valeurs reduites  (de dimension sh_red->nvals)
 */
void hyb_reduc_sum(double *in, double *out, shared_reduc_t *sh_red)
{
  /********************/
  /* Thread Reduction */
  /********************/

  pthread_mutex_lock(sh_red->red_mut);
  {
    // Writting
    for (int i = 0; i < sh_red->nvals; i++)
      {
        sh_red->red_val[i] += in[i];
      }
  }
  pthread_mutex_unlock(sh_red->red_mut);

  // Ensure that output array are null
  for (int i = 0; i < sh_red->nvals; i++)
    {
      out[i] = 0;
    }

  /*****************/
  /* MPI Reduction */
  /*****************/

  // Waiting all thread for avoid mutex on all if and waiting all thread reduction
  pthread_barrier_wait(sh_red->red_bar);

  // one thread can be execute the following code in same time
  sem_wait(sh_red->sem);
  
  if (sh_red->set_master == 0) /* master thread */
    {
      // Set master
      sh_red->set_master = 1;
        
      // Recup MPI data
      int root = 0; // select process 0 to be root
        
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      int size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      // Allocate the buffer
      double *buff = malloc(sizeof(double) * size);

      // Recup in root processus the reduction of all others
      for (int i = 0; i < sh_red->nvals; i++)
        {
          // Need to change with MPI_Reduc
          MPI_Gather(&(sh_red->red_val[i]), 1, MPI_DOUBLE, buff, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

          if (rank == root)
            {
              for (int j = 0; j < size; j++)
                {
                  out[i] += buff[j];
                }
            }
        }

      // Free the buffer
      free(buff);

      // Broadcast to all processus the result
      MPI_Bcast(out, sh_red->nvals, MPI_DOUBLE, root, MPI_COMM_WORLD);

      /*****************/
      /* Thread Update */
      /*****************/

      // Don't need mutex because other thread are waiting (cf: else clause)
      for (int i = 0; i < sh_red->nvals; i++)
        {
          sh_red->red_val[i] = out[i];
        }

      // Finish work
    }
  else
    {
      /*****************/
      /* Thread Update */
      /*****************/

      // Don't need mutex because all thread waiting or finish
      for (int i = 0; i < sh_red->nvals; i++)
        {
          out[i] = sh_red->red_val[i];
        }

      // Finish work
    }

  /***************/
  /* Finish Work */
  /***************/
  sh_red->terminate += 1;

  // Re-init variable for the next call of the function
  if (sh_red->terminate == sh_red->nthreads)
    {
      sh_red->set_master = 0;
      sh_red->terminate = 0;
    }

  // Release semaphore for the next thread
  sem_post(sh_red->sem);

  // Synchronize all thread of all MPI process
  pthread_barrier_wait(sh_red->red_bar);
  MPI_Barrier(MPI_COMM_WORLD);
}


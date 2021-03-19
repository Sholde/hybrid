#include "hyb_reduc.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
void shared_reduc_init(shared_reduc_t *sh_red, int nthreads, int nvals)
{
  /* A COMPLETER */
  sh_red->nvals = nvals;

  sh_red->red_val = malloc(sizeof(double) * nvals);
  memset(sh_red->red_val, 0, sizeof(double) * nvals);
  
  sh_red->red_mut = malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(sh_red->red_mut, NULL);

  sh_red->red_bar = malloc(sizeof(pthread_barrier_t));
  pthread_barrier_init(sh_red->red_bar, NULL, nthreads);

  sh_red->sem = malloc(sizeof(sem_t));
  
  sem_init(sh_red->sem, 0, 0);

  sh_red->terminate = 0;
}

void shared_reduc_destroy(shared_reduc_t *sh_red)
{
  /* A COMPLETER */
  free(sh_red->sem);
  
  pthread_barrier_destroy(sh_red->red_bar);
  free(sh_red->red_bar);

  pthread_mutex_destroy(sh_red->red_mut);
  free(sh_red->red_mut);

  free(sh_red->red_val);
}

/*
 * Reduction  hybride MPI/pthread
 * in  : tableau des valeurs a reduire (de dimension sh_red->nvals)
 * out : tableau des valeurs reduites  (de dimension sh_red->nvals)
 */
void hyb_reduc_sum(double *in, double *out, shared_reduc_t *sh_red)
{
  /* A COMPLETER */
  pthread_mutex_lock(sh_red->red_mut);
  {
    // Writting
    for (int i = 0; i < sh_red->nvals; i++)
      {
        sh_red->red_val[i] += in[i];
      }
  }
  pthread_mutex_unlock(sh_red->red_mut);

  // Waiting result from thread intra-processus
  pthread_barrier_wait(sh_red->red_bar);

  // Updating output array with intra-procesuus reduction
  for (int i = 0; i < sh_red->nvals; i++)
    {
      out[i] = sh_red->red_val[i];
    }

  // MPI Communication
  if (!sh_red->terminate)
    {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      int size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &size);

      int root = 0;

      double *buff = NULL;

      // Allocate the buffer
      buff = malloc(sizeof(double) * size);

      // Recup in root processus the reduction of all others
      for (int i = 0; i < sh_red->nvals; i++)
        {
          MPI_Gather(&(sh_red->red_val[i]), 1, MPI_DOUBLE, buff, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

          if (rank == root)
            {
              printf("here\n");
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

      sh_red->terminate = 1;
      sem_post(sh_red->sem);
    }
  else
    {
      
      sem_wait(sh_red->sem);
      for (int i = 0; i < sh_red->nvals; i++)
        {

        }
      sem_post(sh_red->sem);
    }
}



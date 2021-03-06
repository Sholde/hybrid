#ifndef __HYB_REDUC_H
#define __HYB_REDUC_H

#include <pthread.h>
#include <semaphore.h>

struct shared_reduc_s
{
  /* A COMPLETER */

  int nvals;        /* taille du tableau red_val */
  double *red_val;  /* les valeurs a reduire */

  pthread_mutex_t *red_mut;   /* mutext */
  pthread_barrier_t *red_bar; /* barrier */
  sem_t *sem;                 /* semaphore */

  int set_master;             /* indicate if master thread was choose */
  int terminate;              /* indicate the number of threads that finish their work */
  int nthreads;               /* indicate the number of threads */
};
typedef struct shared_reduc_s shared_reduc_t;

/*
 * Initialisation/destruction d'une structure shared_reduc_t
 * nthreads : nombre de threads (du processus MPI) qui vont participer a la reduction
 * nvals : dimension du tableau a reduire
 */
void shared_reduc_init(shared_reduc_t *sh_red, int nthreads, int nvals);
void shared_reduc_destroy(shared_reduc_t *sh_red);

/*
 * Reduction  hybride MPI/pthread
 * in  : tableau des valeurs a reduire (de dimension sh_red->nvals)
 * out : tableau des valeurs reduites  (de dimension sh_red->nvals)
 */
void hyb_reduc_sum(double *in, double *out, shared_reduc_t *sh_red);

#endif


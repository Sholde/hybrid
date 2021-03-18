#include "hyb_reduc.h"

#include <mpi.h>
#include <stdlib.h>

void shared_reduc_init(shared_reduc_t *sh_red, int nthreads, int nvals)
{
    /* A COMPLETER */
}

void shared_reduc_destroy(shared_reduc_t *sh_red)
{
    /* A COMPLETER */
}

/*
 * Reduction  hybride MPI/pthread
 * in  : tableau des valeurs a reduire (de dimension sh_red->nvals)
 * out : tableau des valeurs reduites  (de dimension sh_red->nvals)
 */
void hyb_reduc_sum(double *in, double *out, shared_reduc_t *sh_red)
{
    /* A COMPLETER */
}



#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <libgen.h>
#include <mpi.h>

#include "mpi_decomp.h"
#include "thr_decomp.h"
#include "hyb_reduc.h"

/*
 * Vecteur
 */
struct vector_s
{
  int N;          /* Vecteur de dimension N */
  double *elt;    /* elt[i] : i-ieme element du vecteur*/
};
typedef struct vector_s vector_t;

void vector_alloc(int N, vector_t *vec)
{
  vec->N = N;
  vec->elt = (double*)malloc(N*sizeof(double));
}

void vector_free(vector_t *vec)
{
  free(vec->elt);
}

/*
 * Initialise le vecteur à 0
 *   " vec = 0 "
 */
void vector_init_0(thr_decomp_t *thr_info, vector_t *vec)
{
  int i;

  for(i = thr_info->thr_ideb; i < thr_info->thr_ifin; i++)
    {
      vec->elt[i] = 0.;
    }
}

/*
 * Multiplie tous les elements du vecteur par un scalaire
 *  " vec *= s "
 */
void vector_mul_scal(thr_decomp_t *thr_info, vector_t *vec, double s)
{
  int i;

  for(i = thr_info->thr_ideb; i < thr_info->thr_ifin; i++)
    {
      vec->elt[i] *= s;
    }
}

/*
 * Affectation d'un vecteur par un autre multiplie' par un scalaire
 *  "  vec_out = s*vec_in  "
 */
void vector_affect_mul_scal(thr_decomp_t *thr_info, vector_t *vec_out, double s, vector_t *vec_in)
{
  int i;

  assert(vec_out->N == vec_in->N);

  for(i = thr_info->thr_ideb; i < thr_info->thr_ifin; i++)
    {
      vec_out->elt[i] = s*vec_in->elt[i];
    }
}

/*
 * Calcule la norme L2 au carre' du vecteur
 *   "  (|| vec ||_2)²  "
 */
double vector_norm2(thr_decomp_t *thr_info, vector_t *vec)
{
  double norm2;
  int i;

  norm2 = 0.;

  for(i = thr_info->thr_ideb; i < thr_info->thr_ifin; i++)
    {
      norm2 += vec->elt[i]*vec->elt[i];
    }

  return norm2;
}

/*
 * Additione à un vecteur un autre vecteur multiplie' par un scalaire
 *  " vec_inout += s*vec_in  "
 */
void vector_add_mul_scal(thr_decomp_t *thr_info, vector_t *vec_inout, double s, vector_t *vec_in)
{
  int i;

  assert(vec_inout->N == vec_in->N);

  for(i = thr_info->thr_ideb; i < thr_info->thr_ifin; i++)
    {
      vec_inout->elt[i] += s*vec_in->elt[i];
    }
}

/*
 * Retourne le rapport de 2 produits scalaires
 *  "  (v1.w1) / (v2.w2)  "
 */
double div_bi_prod_scal(thr_decomp_t *thr_info, vector_t *v1, vector_t *w1, vector_t *v2, vector_t *w2)
{
  int i;

  assert(v1->N == w1->N);
  assert(v1->N == v2->N);
  assert(v1->N == w2->N);

  double scal1, scal2;

  scal1 = 0.;
  scal2 = 0.;

  for(i = thr_info->thr_ideb; i < thr_info->thr_ifin; i++)
    {
      scal1 += v1->elt[i]*w1->elt[i];
      scal2 += v2->elt[i]*w2->elt[i];
    }

  return scal1/scal2;
}

/*
 *  Matrice creuse à 3 bandes
 *  (pour la ligne i, seules les colonnes i-1, i et i+1 sont non nulles)
 */
struct matrix3b_s
{
  int N;          /* Matrice de dimension NxN */

  /* Pour la ligne i, 
   * A(i, i-1) = bnd[0][i]
   * A(i, i)   = bnd[1][i]
   * A(i, i+1) = bnd[2][i]
   * Tous les elements sur les colonnes autres que i-1, i et i+1 sont nuls
   */
  double *bnd[3];
};
typedef struct matrix3b_s matrix3b_t;

void linear_system_alloc_and_init(mpi_decomp_t *mpi_info, matrix3b_t *A, vector_t *vb)
{
  int N = mpi_info->mpi_nloc;
  assert(N > 2);
  int i;

  /* Allocations */
  A->N = N;
  A->bnd[0] = (double*)malloc(N*sizeof(double));
  A->bnd[1] = (double*)malloc(N*sizeof(double));
  A->bnd[2] = (double*)malloc(N*sizeof(double));

  vb->N   = N;
  vb->elt = (double*)malloc(N*sizeof(double));


  /* Remplissage */
  double coeff = 0.01;

  for(i = 0; i < N; i++)
    {
      A->bnd[0][i] = -coeff;
      A->bnd[1][i] = 1. + 2*coeff;
      A->bnd[2][i] = -coeff;

      vb->elt[i] = 1.;
    }

  // First MPI process have the first line of matrix
  if (mpi_info->mpi_rank == 0)
    {
      A->bnd[0][0] = 0.;
      vb->elt[0] = 1. + coeff;
    }

  // Last MPI process have the last line of matrix
  if (mpi_info->mpi_rank == mpi_info->mpi_nproc - 1)
    {
      A->bnd[2][N-1] = 0.;
      vb->elt[N-1] = 1. + coeff;
    }
}

void linear_system_free(matrix3b_t *A, vector_t *vb)
{
  free(A->bnd[0]);
  free(A->bnd[1]);
  free(A->bnd[2]);

  free(vb->elt);
}

/*
 * Produit Matrice-Vecteur
 *  " vy = A.vx  "
 */
void prod_mat_vec(thr_decomp_t *thr_info, vector_t *vy, matrix3b_t *A, vector_t *vx)
{
  assert(A->N == vx->N);
  assert(vy->N == vx->N);

  int i;

  if (mpi_info->mpi_rank == 0)
    {
      if (thr_info->thr_rank == 0)
        {
          /* cas i = 0 */
          i = 0;
          vy->elt[i] = 
            A->bnd[1][i] * vx->elt[i] + 
            A->bnd[2][i] * vx->elt[i+1];
        }

      /* Coeur de la matrice */
      for(i = 1 ; i < A->N-1 ; i++)
        {
          vy->elt[i] = 
            A->bnd[0][i] * vx->elt[i-1] + 
            A->bnd[1][i] * vx->elt[i] + 
            A->bnd[2][i] * vx->elt[i+1];
        }
    }
  else if (mpi_info->mpi_rank == mpi_info->mpi_size - 1)
    {
      if (thr_info->thr_rank == thr_info->nthreads - 1)
        {
          /* cas i = N-1 */
          i = A->N-1;
          vy->elt[i] = 
            A->bnd[0][i] * vx->elt[i-1] + 
            A->bnd[1][i] * vx->elt[i];
        }

      /* Coeur de la matrice */
      for(i = 1 ; i < A->N-1 ; i++)
        {
          vy->elt[i] = 
            A->bnd[0][i] * vx->elt[i-1] + 
            A->bnd[1][i] * vx->elt[i] + 
            A->bnd[2][i] * vx->elt[i+1];
        }
    }
  else
    {
      
    }
  /* cas i = 0 */
  i = 0;
  vy->elt[i] = 
    A->bnd[1][i] * vx->elt[i] + 
    A->bnd[2][i] * vx->elt[i+1];

  /* cas i = N-1 */
  i = A->N-1;
  vy->elt[i] = 
    A->bnd[0][i] * vx->elt[i-1] + 
    A->bnd[1][i] * vx->elt[i];

  /* Coeur de la matrice */
  for(i = 1 ; i < A->N-1 ; i++)
    {
      vy->elt[i] = 
        A->bnd[0][i] * vx->elt[i-1] + 
        A->bnd[1][i] * vx->elt[i] + 
        A->bnd[2][i] * vx->elt[i+1];
    }
}

#define NUM_WORKERS 4

typedef struct args_s
{
  mpi_decomp_t *mpi_info;
  thr_decomp_t *thr_info;
  matrix3b_t *A;
  vector_t *vb;
  vector_t *vx;
} args_t;

/*
 * Algorithme du Gradient Conjugue'
 *   " Resoud le systeme A.vx = vb "
 */
void *gradient_conjugue(void *args_void)
{
  // Recup input
  args_t *args = (args_t *)args_void;
  thr_decomp_t *thr_info = args->thr_info;
  matrix3b_t *A = args->A;
  vector_t *vx = args->vx;
  vector_t *vb = args->vb;
  
  vector_t vg, vh, vw;
  double sn, sn1, sr, sg, seps;
  int k, N;

  assert(A->N == vb->N);
  assert(A->N == vx->N);

  seps = 1.e-12;
  N = A->N;

  vector_alloc(N, &vg);
  vector_alloc(N, &vh);
  vector_alloc(N, &vw);

  /* Initialisation de l'algo */

  vector_init_0(thr_info, vx);
  vector_affect_mul_scal(thr_info, &vg, -1., vb);
  vector_affect_mul_scal(thr_info, &vh, -1., &vg);
  sn = vector_norm2(thr_info, &vg);

  /* Phase iterative de l'algo */

  for(k = 0 ; k < N && sn > seps ; k++)
    {
      printf("Iteration %5d, err = %.4e\n", k, sn);
      prod_mat_vec(thr_info, &vw, A, &vh);

      sr = - div_bi_prod_scal(thr_info, &vg, &vh, &vh, &vw);

      vector_add_mul_scal(thr_info, vx, sr, &vh);
      vector_add_mul_scal(thr_info, &vg, sr, &vw);

      sn1 = vector_norm2(thr_info, &vg);

      sg = sn1 / sn;
      sn = sn1;

      vector_mul_scal(thr_info, &vh, sg);

      vector_add_mul_scal(thr_info, &vh, -1., &vg);
    }

  vector_free(&vg);
  vector_free(&vh);
  vector_free(&vw);
}

/* Verification du resultat
 *  A.vx "doit etre proche" de vb
 */
void *verif_sol(void *args_void)
{
  // Recup input
  args_t *args = (args_t *)args_void;
  thr_decomp_t *thr_info = args->thr_info;
  matrix3b_t *A = args->A;
  vector_t *vx = args->vx;
  vector_t *vb = args->vb;

  vector_t vb_cal;
  double norm2;

  assert(A->N == vb->N);
  assert(A->N == vx->N);

  vector_alloc(A->N, &vb_cal);

  prod_mat_vec(thr_info, &vb_cal, A, vx); /* vb_cal = A.vx */
  vector_add_mul_scal(thr_info, &vb_cal, -1., vb); /* vb_cal = vb_cal - vb */
  norm2 = vector_norm2(thr_info, &vb_cal);

  if (norm2 < 1.e-12)
    {
      printf("Resolution correcte du systeme\n");
    }
  else
    {
      printf("Resolution incorrecte du systeme, erreur : %.4e\n", norm2);
    }
}

/*
  Main
*/
int main(int argc, char **argv)
{
  // Gradient Conjugue
  int N;
  vector_t vx, vb;
  matrix3b_t A;

  // MPI
  int mpi_thread_provided;
  int rank, size;

  // Hybrid
  mpi_decomp_t mpi_info;
  thr_decomp_t thr_info[NUM_WORKERS];
  shared_reduc_t sh_red;

  // Thread
  args_t args[NUM_WORKERS];
  pthread_t pth[NUM_WORKERS];
  
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_thread_provided);
  {
    // Check if MPI_THREAD_SERIALIZED is set
    if (mpi_thread_provided != MPI_THREAD_SERIALIZED)
      {
        printf("Niveau demande' : MPI_THREAD_SERIALIZED, niveau fourni : %d\n",\
               mpi_thread_provided);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 1;
      }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
      {
        // Check argument
        if (argc != 2)
          {
            printf("Usage : %s <N>\n", basename(argv[0]));
            printf("\t<N>    : dimension de la matrice\n");
            abort();
          }
      }
    
    N = atoi(argv[1]);

    mpi_decomp_init(N, &mpi_info);


    shared_reduc_init(&sh_red, NUM_WORKERS, 2); /* 2 = deux valeurs a reduire */

    /* Allocation et construction du systeme lineaire
     */
    linear_system_alloc_and_init(&mpi_info, &A, &vb);
    vector_alloc(mpi_info.mpi_nloc, &vx); /* vx est l'inconnue */


    /* Resolution du systeme lineaire 
     *  A.vx = vb
     * par application de l'algorithme de Gradient Conjugue'
     */
    for (int i = 0; i < NUM_WORKERS; i++)
      {
        thr_decomp_init(mpi_info.mpi_nloc, i, NUM_WORKERS, &(thr_info[i]));

        args[i].mpi_info = &mpi_info;
        args[i].thr_info = &(thr_info[i]);
        args[i].A = &A;
        args[i].vx = &vx;
        args[i].vb = &vb;

        pthread_create(pth + i, NULL, gradient_conjugue, &(args[i]));
      }

    
    for (int i = 0; i < NUM_WORKERS; i++)
      {
        pthread_join(pth[i], NULL);
      }


    /* Verification du resultat
     *  A.vx "doit etre proche" de vb
     */
    for (int i = 0; i < NUM_WORKERS; i++)
      {
        thr_decomp_init(mpi_info.mpi_nloc, i, NUM_WORKERS, &(thr_info[i]));

        args[i].mpi_info = &mpi_info;
        args[i].thr_info = &(thr_info[i]);
        args[i].A = &A;
        args[i].vx = &vx;
        args[i].vb = &vb;

        pthread_create(pth + i, NULL, verif_sol, &(args[i]));
      }

    
    for (int i = 0; i < NUM_WORKERS; i++)
      {
        pthread_join(pth[i], NULL);
      }

    
    /* Liberation memoire
     */
    linear_system_free(&A, &vb);
    vector_free(&vx);
  }
  MPI_Finalize();

  return 0;
}


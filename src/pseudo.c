/*
 *  file :  smo.c
 *  desc :  An implementation of the sequential minimal optimization
 *          algorithm for support vector machines
 *
 *  notes:
 *    (1) what should the initial values of alpha, b be?
 *
 *  refs:
 *    [1]  J.C. Platt. ``Sequential Minimal Optimization: A Fast
 *           Algorithm for Training Support Vector Machines.''
 *           Technical Report MSR-TR-98-14, Microsoft, Seattle,
 *           April 1998.
 *
 *    [2]  A.J. Smola, B. Scholkopf. ``A Tutorial on Support
 *         Vector Regression.''
 *
 *  $Revision: 1.3 $
 */
#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "smo.h"


extern bool_t g_verbose;


/* ----- sample kernels ----- */

/*
 * svm_K_linear( args, x1, x2, d )
 *
 *   Returns the dot product between two d-dimensional points
 *   x1, x2.  `args' is ignored.
 */
real_t
svm_K_linear( void* args, const real_t* x1, const real_t* x2, int d )
{
  int     i;
  real_t  sum;

  const real_t* p1;  /* points to some x1[i] */
  const real_t* p2;  /* points to some x2[i] */

  for( i = 0, sum = 0, p1 = x1, p2 = x2; i < d; i++, p1++, p2++ ) {
    real_t a = *p1;
    real_t b = *p2;
    sum += a*b;
  }

  return sum;
}

/*
 *  svm_K_gaussian( C, x1, x2, d )
 *
 *    Returns:  exp( -C * |x1 - x2|^2 ), a Gaussian (without the
 *  normalizing constant).
 */
real_t
svm_K_gaussian( void* args, const real_t* x1, const real_t* x2, int d )
{
  real_t  C;
  int     i;
  real_t  sum;

  const real_t* p1;  /* points to some x1[i] */
  const real_t* p2;  /* points to some x2[i] */

  /* compute |x1 - x2|^2 */
  for( i = 0, sum = 0, p1 = x1, p2 = x2; i < d; i++, p1++, p2++ ) {
    real_t diff = *p1 - *p2;
    sum += diff*diff;
  }

  C = *((real_t *)args);
  return exp( -C*sum );
}


/*
 *  svm_K_poly( k, x1, x2, d )
 *
 *    Returns:  (x1.x2+1)^k
 */
real_t
svm_K_poly( void* args, const real_t* x1, const real_t* x2, int d )
{
  int     i;
  real_t  sum, prod;
  int     k;

  const real_t* p1;  /* points to some x1[i] */
  const real_t* p2;  /* points to some x2[i] */

  k = *((int *)args);
  if( k == 0 ) return 1;
  assert( k > 0 );

  for( i = 0, sum = 0, p1 = x1, p2 = x2; i < d; i++, p1++, p2++ ) {
    real_t a = *p1;
    real_t b = *p2;
    sum += a*b;
  }

  prod = sum;
  for( i = 1; i < k/2; i *= 2 ) {
    prod *= prod;
  }
  while( i < k ) {
    prod *= sum;
    i++;
  }

  return prod;
}

/*
 *  svm_K_sigmoid( args, x1, x2, d )
 *
 *    Returns:  S( v*x1.x2 - c )
 *  where S(x) = sigmoid applied to x, and v, c are stored in (*args),
 *  which is a pointer to an item of type kernel_sigmoid_t.
 *
 *  QUESTION:  what are the conditions on v,c,x1,x2 s.t. Mercer's
 *  theorem still holds?
 */
real_t
svm_K_sigmoid( void* args, const real_t* x1, const real_t* x2, int d )
{
  int     i;
  real_t  sum, prod;
  int     k;

  const real_t* p1;  /* points to some x1[i] */
  const real_t* p2;  /* points to some x2[i] */

  kernel_sigmoid_t* A = (kernel_sigmoid_t *)args;

  assert( A != NULL );

  for( i = 0, sum = 0, p1 = x1, p2 = x2; i < d; i++, p1++, p2++ ) {
    real_t a = *p1;
    real_t b = *p2;
    sum += a*b;
  }

  sum *= A->v;
  sum += A->c;

  return 1./(1. + exp(sum));
}

/* ----- testing/debugging ----- */

/*
 *  u = svm_eval( S, p )
 *
 *    Evaluate the SVM on a new data point p.  Returns a value that
 *  can be used to determine a label prediction.
 */
real_t
svm_eval( const svm_t* S, const point_t* p )
{
  int i;
  int N;
  int d;

  point_t*  pi;    /* pointer to a training data point */
  real_t*   p_a;   /* pointer to pi's corresponding langrange multiplier */
  real_t    sum;

  kernel_t* K;

  assert( S != NULL );
  assert( S->training != NULL );
  assert( S->training->pts != NULL );
  assert( S->K != NULL );
  assert( S->alpha != NULL );
  assert( p != NULL );

  N = S->training->N;
  d = S->training->dim;

  pi = S->training->pts;
  p_a = S->alpha;
  K = S->K;
  sum = 0;
  for( i = 0; i < N; i++, pi++, p_a++ ) {
    sum += (double)pi->y * (*p_a) * K->func( K->args, pi->x, p->x, d );
  }

  return sum - S->b;
}

/*
 *  svm_dump( S, filename )
 *
 *    Create a debugging dump of the support vector machine.
 *  If filename == NULL, then output goes to stdout.
 */
void
svm_dump( const svm_t* S, const char* filename )
{
  FILE*    fp;
  real_t*  p_a;
  int      i, j;

  point_t* p;
  int      d;

  assert( S != NULL );
  assert( S->training != NULL );
  assert( S->alpha != NULL );

  if( g_verbose ) {
    fprintf( stderr, "--- Dumping the SVM to `%s'...\n",
      filename ? filename : "(stdout)" );
  }

  if( filename == NULL ) {
    fp = stdout;
  } else {
    fp = fopen( filename, "wt" );
    if( fp == NULL ) {
      fprintf( stderr, "!! Warning: can't open dump file `%s' !!\n", filename );
      return;
    }
  }

  d = S->training->dim;

  /* print header */
  fprintf( fp, "%%\n" );
  fprintf( fp, "%% SVM dump file\n" );
  fprintf( fp, "%% format:\n" );
  fprintf( fp, "%%   <dimension> <b> 0 ... 0\n" );
  fprintf( fp, "%%   <alpha_1> <y_1> <x_11> ... <x_1d>\n" );
  fprintf( fp, "%%   <alpha_2> <y_2> <x_21> ... <x_2d>\n" );
  fprintf( fp, "%%   ...\n" );
  fprintf( fp, "%%   <alpha_N> <y_N> <x_N1> ... <x_Nd>\n" );
  fprintf( fp, "%%\n" );

  fprintf( fp, "%d %f", d, S->b );
  for( j = 0; j < d; j++ ) fprintf( fp, " 0" );
  fprintf( fp, "\n" );

  /* print data */
  for( i = 0, p_a = S->alpha, p = S->training->pts;
       i < S->training->N;
       i++, p_a++, p++ ) {
    fprintf( fp, "%f %d", *p_a, p->y );
    for( j = 0; j < d; j++ ) fprintf( fp, " %f", p->x[j] );
    fprintf( fp, "\n" );
  }

  fprintf( fp, "%%\n%% eof\n%%\n" );

  if( g_verbose ) {
    fprintf( fp, "\n" );
  }

  fclose( fp );
}


/* ----- support vector machine stuff ----- */

PRIVATE int
smo_take_step( svm_t* S, int i1, int i2, real_t tol )
{
  point_t*  p1;
  real_t    alpha1;
  label_t   y1;
  real_t    E1;

  point_t*  p2;
  real_t    alpha2;
  label_t   y2;
  real_t    E2;

  kernel_t*       K;
  real_t          k11, k12, k22;
  real_t          eta;

  int   N;
  int   d;
  int   i;

  int   s;

  real_t  C, b;
  real_t  L, H;
  real_t  a1, a2;

  real_t  b1, b2;  /* threshold update */

  assert( S != NULL );

  if( i1 == i2 ) {
    return 0;
  }


  /* init */
  N = S->training->N;
  d = S->training->dim;
  C = S->C;
  b = S->b;

  /* get i1 */
  alpha1 = S->alpha[i1];
  p1 = &(S->training->pts[i1]);
  y1 = p1->y;
  E1 = S->Ecache[i1];

  /* get i2 */
  p2 = &(S->training->pts[i2]);
  alpha2 = S->alpha[i2];
  y2 = p2->y;
  E2 = S->Ecache[i2];

  /* calc L, H */
  s = ((int)y1) * ((int)y2);
  if( s < 0 ) /* y1 != y2 */
  {
    L = MAX( 0, alpha2 - alpha1 );
    H = MIN( C, C + alpha2 - alpha1 );
  }
  else /* y1 != y2 */
  {
    L = MAX( 0, alpha2 + alpha1 - C );
    H = MIN( C, alpha2 + alpha1 );
  }
  if( L == H ) {
    return 0;
  }


  /* evaluate kernels */
  K = S->K;
  k11 = K->func( K->args, p1->x, p1->x, d );
  k12 = K->func( K->args, p1->x, p2->x, d );
  k22 = K->func( K->args, p2->x, p2->x, d );
  eta = k11 + k22 - 2*k12;

  /* compute unconstrained min and then clip to the boundary */
  if( eta > 0 )
  {
    a2 = alpha2 + y2*(E1-E2)/eta;
    if( a2 < L ) a2 = L;
    else if( a2 > H ) a2 = H;
  }
  else  /* eta <= 0 */
  {
    real_t f1 = y1*(E1 + b) - alpha1*k11 - s*alpha2*k12;
    real_t f2 = y2*(E2 + b) - s*alpha1*k12 - alpha2*k22;
    real_t L1 = alpha1 + s*(alpha2 - L);
    real_t H1 = alpha1 + s*(alpha2 - H);
    real_t Lobj = L1*f1 + L*f2 + .5*L1*L1*k11 + .5*L*L*k22 + s*L*L1*k12;
    real_t Hobj = H1*f1 + H*f2 + .5*H1*H1*k11 + .5*H*H*k22 + s*H*H1*k12;

    if( Hobj-Lobj > tol ) a2 = L;
    else if( Lobj-Hobj > tol ) a2 = H;
    else a2 = alpha2;
  }
  if( a2 < 0 ) a2 = 0;
  else if( a2 > C ) a2 = C;


  if( fabs(a2 - alpha2) < tol*(a2+alpha2+tol) ) {
    return 0;
  }

  a1 = alpha1 + s*(alpha2 - a2);
  if( a1 < 0 ) a1 = 0;
  else if( a1 > C ) a1 = C;

  /* did the new alphas (a1, a2) change boundedness? */
  if( (a1 == 0.) || (a1 == C) ) {
    if( S->alpha_bound[i1] == False) {
      S->alpha_bound[i1] = True;
      S->alpha_non_bound--;
    }
  } else { /* not bound */
    if( S->alpha_bound[i1] == True ) {
      S->alpha_bound[i1] = False;
      S->alpha_non_bound++;
    }
  }

  if( (a2 == 0.) || (a2 == C) ) {
    if( S->alpha_bound[i2] == False ) {
      S->alpha_bound[i2] = True;
      S->alpha_non_bound--;
    }
  } else { /* not bound */
    if( S->alpha_bound[i2] == True ) {
      S->alpha_bound[i2] = False;
      S->alpha_non_bound++;
    }
  }

  /* update the threshold */
  b1 = E1 + y1*(a1-alpha1)*k11 + y2*(a2-alpha2)*k12 + b;
  b2 = E2 + y1*(a1-alpha1)*k12 + y2*(a2-alpha2)*k22 + b;

  if( S->alpha_bound[i1] == False ) {
    S->b = b1;
  } else if( S->alpha_bound[i2] == False ) {
    S->b = b2;
  } else { /* both at bounds */
    S->b = .5 * (b1 + b2);
  }

  /* update the error cache */

  for( i = 0; i < N; i++ ) {
    real_t*  x    =  S->training->pts[i].x;
    real_t   k1i  =  K->func( K->args, x, p1->x, d );
    real_t   k2i  =  K->func( K->args, x, p2->x, d );

    S->Ecache[i] +=  y1*(a1 - alpha1)*k1i + y2*(a2 - alpha2)*k2i - S->b + b;
  }

  /* finally, update alphas */
  S->alpha[i1] = a1;
  S->alpha[i2] = a2;

  return 1;
}

PRIVATE int
smo_find_min_error( const svm_t* S )
{
  int       i;
  int       N;
  int       i_min;
  real_t    E_min;
  real_t*   E;
  bool_t*   alpha_bound;

  assert( S != NULL );
  N = S->training->N;

  E = S->Ecache;
  alpha_bound = S->alpha_bound;
  E_min = 0;
  i_min = -1;
  for( i = 0; i < N; i++ ) {
    if( alpha_bound[i] == False )
    {
      real_t E_candidate = E[i];
      if( i_min < 0 ) {
        i_min = i;
        E_min = E_candidate;
      } else if( E_candidate < E_min ) {
        i_min = i;
        E_min = E_candidate;
      } /* if */
    } /* if alpha_bound[i] */
  } /* for */

  return i_min;
}

PRIVATE int
smo_find_max_error( const svm_t* S )
{
  int       i;
  int       N;
  int       i_max;
  real_t    E_max;
  real_t*   E;
  bool_t*   alpha_bound;

  assert( S != NULL );
  N = S->training->N;

  E = S->Ecache;
  alpha_bound = S->alpha_bound;
  E_max = 0;
  i_max = -1;
  for( i = 0; i < N; i++ ) {
    if( alpha_bound[i] == False )
    {
      real_t E_candidate = E[i];
      if( i_max < 0 ) {
        i_max = i;
        E_max = E_candidate;
      } else if( E_candidate > E_max ) {
        i_max = i;
        E_max = E_candidate;
      } /* if */
    } /* if alpha_bound[i] */
  } /* for */

  return i_max;
}

PRIVATE int
smo_examine_example( svm_t* S, int i2, real_t tol )
{
  label_t   y2;
  real_t    alpha2, alpha2_star;
  real_t    C2;
  real_t    E2;
  point_t*  p2;
  real_t    r2;

  int       N;

  assert( S != NULL );

  N = S->training->N;

  p2 = &(S->training->pts[i2]);
  y2 = p2->y;
  alpha2 = S->alpha[i2];
  C2 = S->C;
  E2 = S->Ecache[i2];
  r2 = E2*y2;

  if( (r2 < -tol && alpha2 < C2) || (r2 > tol && alpha2 > 0) )
  {
    int*  i1_list;
    int   i1;
    int   k;

    bool_t* alpha_bound = S->alpha_bound;

    #if 0
      fprintf( stderr, "  @@   i2 = %d, alph2 = %g,"
                       " E2 = %f, r2 = %f, non_bound = %d\n",
               i2, alpha2, E2, r2, S->alpha_non_bound );
    #endif

    if( S->alpha_non_bound > 1 ) {
      if( E2 > 0 )
        i1 = smo_find_min_error( S );
      else /* E2 <= 0 */
        i1 = smo_find_max_error( S );

      if( i1 >= 0 ) {
        if( smo_take_step(S, i1, i2, tol) )
          return 1;
      }
    } /* if */

    i1_list = S->i1_randlist;
    init_randperm( i1_list, N );
    for( k = 0; k < N; k++ ) {
      i1 = i1_list[k];

      if( alpha_bound[i1] == False ) {
        if( smo_take_step(S, i1, i2, tol) )
          return 1;
      }
    } /* for */

    init_randperm( i1_list, N );
    for( k = 0; k < N; k++ ) {
      i1 = i1_list[k];
      if( smo_take_step(S, i1, i2, tol) )
        return 1;
    } /* for */
  } /* if alpha2 in bounds */

  return 0;
} /* smo_examine_example */

svm_t *
svm_smo_new( points_t* data, real_t tol, kernel_t* kernel, real_t C )
{
  int N;                 /* size of data set */

  svm_t*   S;           /* new SVM */
  real_t*  alpha;       /* == S->alpha */
  bool_t*  alpha_bound; /* == S->alpha_bound */

  int numChanged;
  bool_t examineAll;
  int LoopCounter;
  int MinimumNumChanged;

  int i;

  /* extract some parameters for convenience */
  N = data->N;


  /* alloc */
  S = (svm_t *)malloc( sizeof(svm_t) );
  assert( S != NULL );

  S->K = kernel;
  S->C = C;
  S->training = data;


  /* initialize alphas */
  S->alpha = (real_t *)malloc( sizeof(real_t)*N );
  S->alpha_bound = (bool_t *)malloc( sizeof(bool_t)*N );
  assert( (S->alpha != NULL) );

  alpha = S->alpha;
  alpha_bound = S->alpha_bound;
  S->alpha_non_bound = 0;
  for( i = 0; i < N; i++ )
  {
    alpha[i] = 0;
    S->alpha_bound[i] = True;
  }


  /* initialize the error caches */
  S->Ecache = (real_t *)malloc( sizeof(real_t)*N );
  assert( (S->Ecache != NULL) );

  S->b = 0;
  for( i = 0; i < N; i++ )
    S->Ecache[i] = -S->training->pts[i].y;


  /* init misc */
  S->i1_randlist = (int *)malloc( sizeof(int)*N );
  assert( S->i1_randlist != NULL );
  for( i = 0; i < N; i++ ) {
    S->i1_randlist[i] = i;
  }


  /* main loop */
  numChanged = 0;
  examineAll = True;
  LoopCounter = 0;

  while( (numChanged > 0) || (examineAll == True) )
  {
    LoopCounter++;
    numChanged = 0;
    if( examineAll == True )
    {
      for( i = 0; i < N; i++ ) {
        numChanged += smo_examine_example( S, i, tol );
      } /* for */
    }
    else /* examineAll == False */
    {
      for( i = 0; i < N; i++ ) {
        if( alpha_bound[i] == False )
        {
          numChanged += smo_examine_example( S, i, tol );
        }
      } /* for */
    } /* if */

    if( examineAll == True )
      examineAll = False;
    else if( numChanged == 0 )
      examineAll = True;

     if( g_verbose && (((LoopCounter-1)%10) == 0) ) {
       fprintf( stderr, "[%d] b: %f, non-bound: %d\n",
         LoopCounter, S->b, S->alpha_non_bound );
     }
  } /* while */


  /* done; clean-up and return */

  free( S->i1_randlist );
  S->i1_randlist = NULL;

  return S;
}



void
svm_smo_delete( svm_t* S )
{
  if( S != NULL ) {
    if( S->Ecache != NULL ) free( S->Ecache );
    if( S->alpha_bound != NULL ) free( S->alpha_bound );
    if( S->i1_randlist != NULL ) free( S->i1_randlist );
    if( S->alpha != NULL ) free( S->alpha );
  }
}

/* ----- misc functions ----- */

/*
 * init_randperm( A, n )
 *
 *   Initializes A[0..n-1] with a random shuffle of the numbers 0..n-1.
 */
void
init_randperm( int* A, int n )
{
  int i;
  int x;

  for( i = 0; i < n; i++ ) {
    A[i] = i;
  }

  for( i = 0; i < n; i++ ) {
    int temp;

    /* swap A[i] with a random A[x] */
    x = lrand48() % n;
    temp = A[i];
    A[i] = A[x];
    A[x] = temp;
  }
}

/*
 * $Log: smo.c,v $
 * Revision 1.3  1999/05/20 10:14:26  richie
 * *** empty log message ***
 *
 * Revision 1.2  1999/05/18 22:21:48  richie
 * cleaned up miscellaneous pre-processor junk
 *
 * Revision 1.1  1999/05/18 22:20:07  richie
 * Initial revision
 *
 *
 * eof */

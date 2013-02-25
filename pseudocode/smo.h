/*
 *  file :  smo.h
 *  desc :  An implementation of the sequential minimal optimization
 *          algorithm for support vector machines
 *
 *  refs:
 *    [1]  Platt, J.C. "Sequential Minimal Optimization: A Fast
 *           Algorithm for Training Support Vector Machines."
 *           Technical Report MSR-TR-98-14, Microsoft, Seattle,
 *           April 1998.
 *
 *    [2]  A.J. Smola, B. Scholkopf. ``A Tutorial on Support
 *         Vector Regression.''
 *
 *  $Revision: 1.2 $
 */

#ifndef INC_SMO_H

#define MAX(a,b)  ((a)>(b)?(a):(b))
#define MIN(a,b)  ((a)<(b)?(a):(b))

#define PRIVATE static


/* ----- data types ----- */

typedef enum { False = 0, True = 1 } bool_t;   /* ad hoc boolean type */
typedef float real_t;    /* input data type */

/* i didn't use enumerated type here because of weird Sun cc/enum-int
 * conversion that doesn't quite do what i expect
 */
#ifdef AIX_RS6K
  typedef int _label_t;
  #define label_t _label_t
#else
  typedef int label_t;
#endif
#define CL_ONE 1
#define CL_NONE 0
#define CL_TWO (-CL_ONE)


/* single data point */
typedef struct tag_point_t {
  real_t*  x;        /* data point */
  label_t  y;        /* class label */
} point_t;

/* collection of data points
 *   pts undefined if N <= 0
 */
typedef struct tag_points_t {
  int dim;       /* dimension of data stored within */
  int N;         /* number of data points */
  point_t* pts;  /* actual data points */
} points_t;



/*
 * Kernel function
 *   K( args, x_1, x_2, d ) should be a kernel function that operates on two
 *   d-dimensional data points x_1 and x_2.
 */
typedef  real_t (*func_kernel_t)( void* args, const real_t*, const real_t*,
                                  int d );
typedef  struct tag_kernel_t {
  func_kernel_t  func;   /* pointer to the actual kernel function */
  void*          args;   /* pointer to args (see individual kernels) */
} kernel_t;

/* Support vector machine */
typedef struct tag_svm_t {
  points_t*      training;    /* training points */
  kernel_t*      K;           /* kernel function */
  real_t*        alpha;       /* lagrange multipliers */
  real_t         b;           /* threshold */

  /* for the SMO algorithm */
  int C;         /* bounding box parameter */

  real_t*        Ecache;      /* error cache */

  int            alpha_non_bound;   /* number of non-bound examples */
  bool_t*        alpha_bound; /* is alpha bound? */

  /* temporary stuff */
  int*  i1_randlist;   /* for storing random permutations */

} svm_t;



/* ----- function prototypes ----- */

/* ----- sample kernels ----- */
real_t svm_K_linear( void* args, const real_t* x1, const real_t* x2, int d );
  /*
   *   Returns the dot product between two d-dimensional points
   *   x1, x2.  `args' is ignored.
   */

real_t svm_K_gaussian( void* args, const real_t* x1, const real_t* x2, int d );
  /*
   *  Returns:  exp( C * |x1 - x2|^2 ), a Gaussian (without the
   *  normalizing constant).  `args' is assumed to be a single
   *  real_t scalar C.
   */

  #define DEF_KERNEL_GAUSSIAN_C  0.5


real_t svm_K_poly( void* args, const real_t* x1, const real_t* x2, int d );
  /*                       k
   *  Returns:  (x1.x2 + 1)    i.e., polynomial with degree d
   *  args is assumed to be a single integer k.
   */

  #define DEF_KERNEL_POLY_DEGREE  2



real_t svm_K_sigmoid( void* args, const real_t* x1, const real_t* x2, int d );
  /*
   *  Returns:  S( v*x1.x2 - c )
   *  where S(x) = sigmoid applied to x, and v, c are stored in (*args),
   *  which is a pointer to an item of type kernel_sigmoid_t.
   *
   *  WARNING:  The sigmoid function S(x) does not satisfy Mercer's
   *  condition unless v*x1.x2 - c <= 0
   */

  /* argument structure for the sigmoid kernel */
  typedef struct tag_kernel_sigmoid_t {
    real_t v;
    real_t c;
  } kernel_sigmoid_t;

  #define DEF_KERNEL_SIGMOID_V  1
  #define DEF_KERNEL_SIGMOID_C  0


/* ----- support vector machine stuff ----- */
real_t svm_eval( const svm_t* S, const point_t* p );
  /*
   *  Evaluate the SVM on a new data point p.  Returns a value that
   *  can be used to determine a label prediction.
   */

void svm_dump( const svm_t* S, const char* filename );
  /*
   *  Create a debugging dump of the support vector machine.
   *  If filename == NULL, then output goes to stdout.
   */

svm_t * svm_smo_new( points_t* data, real_t tol, kernel_t* K, real_t C );
  /*
   *  Creates a new support vector machine S, and trains it on the
   *  points given by data.  Uses the SMO algorithm described in [1],[2].
   *  Uses a convergence tolerance of `tol', the kernel `kernel', and
   *  box constraint C (for possibly overlapping data).
   *
   *  Returns a pointer S to the new SVM.
   */


/* ----- misc functions ----- */

void init_randperm( int* A, int n );
  /*
   *   Initializes A[0..n-1] with a random shuffle of the numbers 0..n-1.
   */

#endif

/*
 *  $Log: smo.h,v $
 *  Revision 1.2  1999/05/20 10:14:26  richie
 *  *** empty log message ***
 *
 *  Revision 1.1  1999/05/18 22:20:07  richie
 *  Initial revision
 *
 *
 * eof */

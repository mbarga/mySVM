/*
 *  file :  main.c
 *  desc :  Driver/test program for my implementation of the
 *          sequential minimal optimization algorithm for
 *          support vector machines.
 *
 *  $Revision: 1.2 $
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "smo.h"

/* ----- program parameters ----- */
char*     g_progname;         /* program's command-line name */
char*     g_inputfile;        /* training data file */
char*     g_dumpfile;         /* SVM dump file */
char*     g_testfile;         /* SVM validation log file */
float     g_valid_frac;       /* fraction of points to use for training */
real_t    g_smo_tol;          /* SMO algorithm tolerance */
real_t    g_C;                /* initial box dimension */
bool_t    g_validate;         /* validate results */
bool_t    g_validate_append;  /* append validation results */
kernel_t  g_kernel;           /* which SVM kernel */
int       g_verbose;          /* verbosity level (0 == quiet) */
long      g_seed;             /* a random seed */



/*
 * usage()
 *   Prints a usage error message to the user.
 */
void
usage( void )
{
  fprintf( stderr, "\nusage: %s", g_progname );
  fprintf( stderr, " [-v]"
                   " [-k <kernel>]"
                   " [-t <tol>]"
                   " [-f <infile>]"
                   " [-d <outfile>]"
                   " [-x[a] <frac> [logfile]]"
                   " [-seed [<val>]"
                   "\n" );
  fprintf( stderr, "where:\n" );
  fprintf( stderr, "  -v :  turn on verbose mode\n" );
  fprintf( stderr, "  -k {linear | gaussian [<c>] | poly [<d>]\n"
                   "       | sigmoid [<v> <c>]}\n"
                   "     :  select kernel to use\n" );
  fprintf( stderr, "  -t <tol> :  select convergence tolerance (real)\n" );
  fprintf( stderr, "  -C <box> :  select box constraint value (real)\n" );
  fprintf( stderr, "  -f <infile> :  specifies input data file\n" );
  fprintf( stderr, "  -d <dumpfile> :  dump the SVM to a file\n" );
  fprintf( stderr, "  -x[a] <frac> [logfile] :  validation test (a to append)\n" );
  fprintf( stderr, "  -seed [<seed>] :  set the random seed (note: currently,\n"
                   "     this only disables automatic random seeding)\n" );
  fprintf( stderr, "\n" );
}

void
parse_cmd_line( int argc, char* argv[] )
{
  int i;

  char*  sum_kernel;

  /* initialize return parameters to default values */
  g_progname = strdup( argv[0] );
  g_inputfile = NULL;   /* none specified */
  g_dumpfile = NULL;
  g_testfile = NULL;
  g_valid_frac = .9;
  g_smo_tol = .001;
  g_C = 1;
  g_kernel.func = svm_K_linear;  g_kernel.args = NULL;
  sum_kernel = "linear";
  g_verbose = 0;
  g_validate = False;
  g_validate_append = False;
  g_seed = -1;

  i = 1;
  while( i < argc ) {
    if( strcmp(argv[i], "-f") == 0 ) {
      g_inputfile = strdup( argv[++i] );
    } else if( strcmp(argv[i], "-d") == 0 ) {
      g_dumpfile = strdup( argv[++i] );
    } else if( strcmp(argv[i], "-x") == 0 ) {
      sscanf( argv[++i], "%f", &g_valid_frac );
      if( (i < (argc-1)) && (argv[i+1][0] != '-') ) {
        g_testfile = strdup( argv[++i] );
      }
      g_validate = True;
      g_validate_append = False;
    } else if( strcmp(argv[i], "-xa") == 0 ) {
      sscanf( argv[++i], "%f", &g_valid_frac );
      if( (i < (argc-1)) && (argv[i+1][0] != '-') ) {
        g_testfile = strdup( argv[++i] );
      }
      g_validate = True;
      g_validate_append = True;
    } else if( strcmp(argv[i], "-t") == 0 ) {
      sscanf( argv[++i], "%f", g_smo_tol );
    } else if( strcmp(argv[i], "-C") == 0 ) {
      sscanf( argv[++i], "%f", &g_C );
    } else if( strcmp(argv[i], "-k") == 0 ) {
      ++i;
      switch( argv[i][0] ) {
        case 'l':
        case 'L':
          g_kernel.func = svm_K_linear;
          g_kernel.args = NULL;
          sum_kernel = "linear";
          break;

        case 'g':
        case 'G':
          g_kernel.func = svm_K_gaussian;
          g_kernel.args = (real_t *)malloc( sizeof(real_t) );
          *((real_t *)g_kernel.args) = DEF_KERNEL_GAUSSIAN_C;
          if( (i+1) < argc ) {
            if( argv[i+1][0] != '-' ) {
              sscanf( argv[++i], "%f", (real_t *)g_kernel.args );
            }
          }
          sum_kernel = "gaussian";
          break;

        case 'p':
        case 'P':
          g_kernel.func = svm_K_poly;
          g_kernel.args = (int *)malloc( sizeof(int) );
          *((real_t *)(g_kernel.args)) = DEF_KERNEL_POLY_DEGREE;
          if( (i+1) < argc ) {
            if( argv[i+1][0] != '-' ) {
              sscanf( argv[++i], "%d", (real_t *)g_kernel.args );
            }
          }
          sum_kernel = "polynomial";
          break;

        case 's':
        case 'S':
          g_kernel.func = svm_K_sigmoid;
          g_kernel.args = (kernel_sigmoid_t *)malloc(sizeof(kernel_sigmoid_t));
          ((kernel_sigmoid_t *)g_kernel.args)->v = DEF_KERNEL_SIGMOID_V;
          ((kernel_sigmoid_t *)g_kernel.args)->c = DEF_KERNEL_SIGMOID_C;
          if( (i+1) < argc ) {
            if( argv[i+1][0] != '-' ) {
              sscanf( argv[++i], "%d",
                      &((kernel_sigmoid_t *)g_kernel.args)->v );
              sscanf( argv[++i], "%d",
                      &((kernel_sigmoid_t *)g_kernel.args)->c );
            }
          }
          sum_kernel = "sigmoid";
          break;

        default:
          fprintf( stderr, "!!! unrecognized kernel: `%s' !!!\n", argv[i] );
          usage();
          exit(2);
      }
    } else if( strcmp(argv[i], "-seed") == 0 ) {
      if( (i+1) < argc )
        if( argv[i+1][0] != '-' ) {
          sscanf( argv[++i], "%d", &g_seed );
        }
    } else if( strcmp(argv[i], "-v") == 0 ) {
      g_verbose = 1;
    } else if( strcmp(argv[i], "-h") == 0 ) {
      usage();
      exit(0);
    } else {  /* unrecognized command-line option */
      fprintf( stderr, "!!! unrecognized command line option `%s' !!!\n", argv[i] );
      usage();
      exit(1);
    }

    i++;
  }

  if( g_inputfile == NULL ) {
    fprintf( stderr, "!!! no input data file specified !!!\n" );
    usage();
    exit(1);
  }

  /* command-line parameter summary */
  if( g_verbose ) {
    fprintf( stderr, "\n--- parameter summary ---\n" );
    fprintf( stderr, "  verbosity: on\n" );
    fprintf( stderr, "  kernel: %s\n", sum_kernel );
    fprintf( stderr, "  SMO tolerance: %f\n", g_smo_tol );
    fprintf( stderr, "  C: %f\n", g_C );
    fprintf( stderr, "  data file: %s\n", g_inputfile );
    fprintf( stderr, "  dump file: %s\n",
      g_dumpfile ? g_dumpfile : "(stdout)" );
    fprintf( stderr, "  validation: %s (%f%%)\n",
      g_validate ? "on" : "off", g_valid_frac*100.0 );
    if( g_testfile )
      fprintf( stderr, "    --> %s\n",
        g_testfile ? g_testfile : "(stdout)" );
    fprintf( stderr, "\n" );
  }
}

/*
 *  y = read_data( filename );
 *
 *    Returns a set of points read in from `filename'.  The format of
 *  the ASCII input file is (line numbers shown in [#]):
 *
 *    [0]     dimension  number_of_points
 *    [1]     y1 x1_1 x1_2 ... x1_d
 *    [2]     y2 x2_1 x2_2 ... x2_d
 *    ...
 *    [n]     yn xn_1 xn_2 ... xn_d
 */
points_t *
read_data( const char* filename )
{
  FILE* fp;
  points_t*  new_points;
  point_t*   new_p;
  int i;
  char buf[BUFSIZ];
  int  linecount;

  if( g_verbose ) {
    fprintf( stderr, "--- opening input data file `%s'...\n", filename );
  }

  /* open input file and read the header */
  fp = fopen( filename, "rt" );
  if( fp == NULL ) {
    fprintf( stderr, "!!! error opening data file: `%s' !!!\n", filename );
    exit(3);
  }
  linecount = 0;

  /* alloc preliminary space for points */
  new_points = (points_t *)malloc( sizeof(points_t) );
  assert( new_points != NULL );

  /* read in the header */
  if( fscanf(fp, "%d %d", &(new_points->dim), &(new_points->N)) == 2 ) {
    linecount++;
  } else {
    fprintf( stderr, "*** Can't read header for `%s' ***\n", filename );
    assert( 0 );
  }
  assert( (new_points->dim > 0) && (new_points->N > 0) );
  new_points->pts = (point_t *)malloc( sizeof(point_t)*(new_points->N) );
  assert( new_points->pts != NULL );

  if( g_verbose ) {
    fprintf( stderr, "  (reading %d %d-dimensional points...)\n",
      new_points->N, new_points->dim );
  }

  /* read in data points */
  new_p = new_points->pts;
  for( i = 0; i < new_points->N; i++ ) {
    float y_float;
    int d;

    /* read label */
    fscanf( fp, "%f", &y_float );
    new_p->y = (int)y_float;

    if( (new_p->y != 1) && (new_p->y != -1) ) {
      fprintf( stderr, "*** grak:  bad label (i=%d)\n", i );
      assert( (new_p->y == 1) || (new_p->y == -1) );
    }

    /* read data point */
    new_p->x = (real_t *)malloc( sizeof(real_t)*(new_points->dim) );
    assert( new_p != NULL );
    for( d = 0; d < new_points->dim; d++ ) {
      fscanf( fp, "%f", &(new_p->x[d]) );
    }

    new_p++;
  }

  /* done */
  if( g_verbose ) {
    fprintf( stderr, "\n" );
  }
  fclose( fp );
  return new_points;
}


void
shuffle_data( points_t* x )
{
  int N;
  int i;
  point_t*  p;

  assert( x != NULL );

  N = x->N;

  p = x->pts;
  for( i = 0; i < N; i++ ) {
    int new_loc = lrand48() % N;
    real_t* x_old;
    label_t y_old;

    /* swap any point to any neighbor */

    x_old = p[i].x;
    y_old = p[i].y;
    p[i].x = p[new_loc].x;
    p[i].y = p[new_loc].y;
    p[new_loc].x = x_old;
    p[new_loc].y = y_old;
  }
}


/*
 *  split_data( x, f, y1, y2 );
 *
 *    Split the list of points x into two sublists y1, y2 such that
 *  the number of points in y1 is a fraction f of the number of points
 *  in x.
 *
 *  pre :  # of points in x >= 1
 *
 *  NOTE:  the points in y1 and y2 are shallow copies of x.
 *         y1 will always have at least 1 point.
 */
void
split_data( const points_t* x, float f, points_t* y1, points_t* y2 )
{
  int N;
  int d;

  int N1;
  int N2;

  assert( x != NULL );
  assert( y1 != NULL );
  assert( y2 != NULL );
  assert( x->N >= 1 );

  N = x->N;
  d = x->dim;
  N1 = (int)((float)N*f+.5);
  N2 = N - N1;

  assert( N1 >= 1 );
  assert( N2 >= 0 );

  y1->N = N1; y1->dim = d;
  y2->N = N2; y2->dim = d;

  y1->pts = x->pts;
  y2->pts = &(x->pts[N1]);
}

/*
 *  test_svm( S, x, filename );
 *
 *    Evaluates the SVM S on all the data points x, and computes
 *  the error rate.  All output dumped to the file `filename'.
 *  If filename == NULL, output goes to stdout.
 *
 */
void
test_svm( const svm_t* S, const points_t* x, const char* filename )
{
  FILE*     fp;
  bool_t    pass;
  int       i, j, N, d;
  point_t*  p;
  int       N_err;

  assert( S != NULL );
  assert( x != NULL );

  if( g_verbose ) {
    fprintf( stderr, "--- validating the SVM ---\n" );
  }

  if( filename == NULL ) {
    fp = stdout;
  } else {
    if( g_validate_append )
      fp = fopen( filename, "wt" );
    else
      fp = fopen( filename, "at" );
    if( fp == NULL ) {
      fprintf( stderr, "!!! can't open test log file `%s' !!!\n", filename );
      return;
    }
  }

  N = x->N;
  N_err = 0;
  d = x->dim;
  for( p = x->pts, i = 0 ; i < N; i++, p++ ) {
    label_t  y;
    real_t   u;

    u = svm_eval( S, p );

    for( j = 0; j < d; j++ ) {
      fprintf( fp, "%f ", p->x[j] );
    }
    fprintf( fp, "%f\n", u );

    if( u > 0 ) {
      y = CL_ONE;
    } else {
      y = CL_TWO;
    }

     if( y != p->y ) N_err++;
  }

  fprintf( fp, "Validation:  %d / %d (%f%%)\n",
    N - N_err, N, (float)(N-N_err)/N*100.0 );

  if( g_verbose && filename != NULL ) {
    fprintf( stderr, "Validation:  %d / %d (%f%%)\n",
      N - N_err, N, (float)(N-N_err)/N*100.0 );
  }

  fclose( fp );

  if( g_verbose ) {
    fprintf( stderr, "\n" );
  }
}

/* ------------------------------------------------------------ */
int
main( int argc, char* argv[] )
{
  points_t*  data;   /* raw data */
  svm_t*     svm;    /* support vector machine */
  points_t   train;  /* training data set */
  points_t   val;    /* validation set */

  /* set-up */
  parse_cmd_line( argc, argv );
  data = read_data( g_inputfile );

  if( g_seed < 0 ) {
    unsigned short* seedbuf = (unsigned short *)malloc(sizeof(unsigned short)*3);
    *seedbuf = (unsigned short)time(NULL);
    seed48( seedbuf );
  }

  /* create the SVM and validate it */
  shuffle_data( data );
  split_data( data, g_valid_frac, &train, &val );

  if( g_verbose ) fprintf( stderr, "\n--- building the SVM ---\n" );
  svm = svm_smo_new( &train, g_smo_tol, &g_kernel, g_C );

  if( g_verbose ) fprintf( stderr, "\n--- saving the SVM ---\n" );
  svm_dump( svm, g_dumpfile );

  /* test the SVM */
  test_svm( svm, &val, g_testfile );

  /* done! */
  fprintf( stderr, "\n" );
  return 0;
}

/*
 * $Log: main.c,v $
 * Revision 1.2  1999/05/20 10:14:26  richie
 * *** empty log message ***
 *
 * Revision 1.1  1999/05/18 22:20:07  richie
 * Initial revision
 *
 *
 * eof */

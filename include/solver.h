/**
 * \brief Library containing methods for updating the SVM structure
 *
 * This is a paragraph of text for the solver header
 *
 */
#ifndef _SOLVER_H
#define _SOLVER_H 

#include <time.h>
#include <cache.h>

#define C 0.2			// TODO: slack?
#define EPS 0.0001		// 10^-3
#define N 3 			// number of examples
#define M 3 			// number of features
namespace MySVM {

class Solver {
private:

	// all these arrays need to know the size in order to run w/o seg fault
	double y[N];
	double x[N][M]; // training examples (and their associated features)
	double b[N];

	/** \brief 'TakeStep' Optimize the SVM for a pair of alphas
	 * 	\param index_i index of first alpha of the pair being analyzed
	 * 	\param index_j index of second alpha of the pair being analyzed
	 * 	\return An integer indicating whether anything was updated ('0' if result was within the error threshold, i.e. nothing changed)
	 */
	int update(int index_i, int index_j);

	/**	\brief Evaluates the kernel function on two inputs
	 * 	\return Evaluated dot product
	 */
	double kernel(double *, double *);

public:
	double alpha[N];

	/** \brief 'ExamineExample' Checks if SVM structure satisfies KKT conditions; If for a given index the conditions are not met, calls update() to optimize for current alpha pair
	 * 	\param index index to check
	 * 	\return Returns '1' if anything was updated
	 */
	int examine(int index);

	Solver();

};// end Solver

}
;// namespace
#endif

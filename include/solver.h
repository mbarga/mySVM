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

#define C 2
#define EPS 0.01

namespace MySVM {

class Solver {
private:
	/** \brief 'TakeStep' Optimize the SVM for a pair of alphas
	 * 	\param index_i index of first alpha of the pair being analyzed
	 * 	\param index_j index of second alpha of the pair being analyzed
	 * 	\return An integer indicating whether anything was updated ('0' if result was within the error threshold, i.e. nothing changed)
	 */
	int update(int index_i, int index_j);

public:
	double *y;		//[N];
	double **x;		//[N][M];
	double *alpha; 	//[N]
	double *w; 		//[M]
	double b;
	double *error; 	//[N];
	double length;
	double features;
	int *randi;

	/** \brief 'ExamineExample' Checks if SVM structure satisfies KKT conditions; If for a given index the conditions are not met, calls update() to optimize for current alpha pair
	 * 	\param index index to check
	 * 	\return Returns '1' if anything was updated
	 */
	int examine(int index);

	/**	\brief Evaluates the kernel function on two inputs
	 * 	\return Evaluated dot product
	 */
	double kernel(double* x[] , int, int);

	Solver();
	void randperm( int*, int);
	void print();

};// end Solver

}
;// namespace
#endif

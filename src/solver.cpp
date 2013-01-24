#include <mysvm.h>
#include <solver.h>

#define getMax(a,b) a>b?a:b
#define getMin(a,b) a<b?a:b

namespace MySVM {

// Solver class constructor
Solver::Solver() {

	srand(time(NULL));

	for (int example_i = 0; example_i < N; example_i++) {
		// private
		//TODO randomize y's for simplicity now
		y[example_i] = rand() % RAND_MAX;
		for (int feature_i = 0; feature_i < M; feature_i++) {
			x[example_i][feature_i] = rand() % 100;
			w[feature_i] = 0;
		}

		//public
		alpha[example_i] = 0;
	}

	b = 0;
}

double Solver::kernel(double *x1, double *x2) {

	int dotProduct = 0;

	for (int i = 0; i < M; ++i) {
		dotProduct += (x1[i] * x2[i]);
	}

	printf("Value of the linear kernel was: %d\n", dotProduct);
	return dotProduct;
}

int Solver::examine(int index_j) {

	double y2 = y[index_j];
	double alph2 = alpha[index_j];
	double E2 = 1; // svm output[i2] - y2
	double r2 = E2 * y2;
	int numAlphaNotAtBounds = 0;
	int nonBoundValueAlphaIndices[N] = { 0 };

	// TEST
	printf("EXAMINING\n");

	int index_i = 0;
	if (((r2 < -EPS) && (alph2 < C)) || ((r2 > EPS) && (alph2 > 0))) {

		// find number and indices of non-zero and non-C alphas
		for (index_i = 0; index_i < N; index_i++) {
			if ((abs(alpha[index_i]) > EPS)
					&& ((abs(alpha[index_i] < (C - EPS))) || (abs(
							alpha[index_i]) > (C + EPS)))) {
				//TODO: grow this array dynamically
				nonBoundValueAlphaIndices[numAlphaNotAtBounds] = alpha[index_i];
				numAlphaNotAtBounds++;
			}
		}

		// lookat tinySVM
		/*
		 if ( numAlphaNotAtBounds > 1 ) {
		 // perform second choise heuristic to choose index_i
		 index_i = 0; // TODO: choose multiplier to maximize the step taken; i.e. max(|E1 - E2|);
		 if ( update(index_i, index_j) ) {
		 return 1;
		 }
		 }
		 */

		/*
		 // else loop over all non-zero and non-c alpha, starting at a random point
		 // TODO: start at random point
		 for(int currentAlphaIndex = 0; currentAlphaIndex < numAlphaNotAtBounds; currentAlphaIndex++) {
		 index_i = nonBoundValueAlphaIndices[currentAlphaIndex];
		 if ( update(index_i, index_j) ) {
		 return 1;
		 }
		 }
		 */

		/*
		 // else loop over all possible i1, starting at random point
		 // TODO: start at random point
		 for(index_i = 0; index_i < N; index_i++) {
		 if ( update(index_i, index_j) ) {
		 return 1;
		 }
		 }
		 */
	}// if error > tolerance

	return 0; // no changes made
}// solver::examine

int Solver::update(int index_i, int index_j) {

	if (index_i == index_j) {
		return 0;
	}

	double y1 = y[index_i];
	double y2 = y[index_j];
	double alph1 = alpha[index_i];
	double alph2 = alpha[index_j];
	double E1 = 1; // svm output[i] - y1
	double s = y1 * y2;
	double a1 = 0;
	double a2 = 0;

	// TODO: fix
	double E2 = 0;

	// compute L and H via equations
	double H = 0;
	double L = 0;
	if (s < 0) {
		L = getMax(0, (alph2-alph1));
		H = getMin(C, (C+alph2-alph1));
	} else {
		L = getMax(0, (alph2+alph1-C));
		H = getMin(C, (alph2+alph1));
	}

	if (L == H) {
		return 0;
	}

	double k11 = kernel(x[index_i], x[index_i]);//<x1,x1>;
	double k12 = kernel(x[index_i], x[index_j]);//<x1,x2>;
	double k22 = kernel(x[index_j], x[index_j]);//<x2,x2>;
	double eta = k11 + k22 - (2 * k12);

	if (eta > 0) {
		a2 = alph2 + (y2 * (E1 - E2) / eta);
		if (a2 < L) {
			a2 = L;
		} else if (a2 > H) {
			a2 = H;
		}

	} else {
		//TODO: calculate these objectives
		double Lobj = (y2 * L * x[]) - b; //objective function at a2 = L;
		double Hobj = (y2 * H * x[]) - b; //objective function at a2 = H;

		if (Lobj < (Hobj - EPS)) {
			a2 = L;
		} else if (Lobj > (Hobj + EPS)) {
			a2 = H;
		} else {
			a2 = alph2;
		}
	}

	if (abs(a2 - alph2) < EPS * (a2 + alph2 + EPS)) {
		return 0;
	}

	a1 = alph1 + s * (alph2 - a2);

	// update bias (threshold) to reflect change in alphas
	// 2.3 Computing the Threshold
	double b1 = E1 + y1*((a1 - alph1) * k11) + y2*((a2 - alph2) * k12) + b;
	double b2 = E2 + y1*((a1 - alph1) * k12) + y2*((a2 - alph2) * k22) + b;
	if (((a1 == H) | (a1 == L)) & ((a2 == H) | (a2 == L))) { // could this be simplified?
		b = (b1 + b2) / 2;
	} else {
		// alphas not at bounds: b1 should be equal to b2
		b = b1;
	}

	// update weight vector
	// 2.4 An Optimization for Linear SVMs
	for (int findex = 0; findex < M; findex++) {
		w[findex] = w[findex] + y1*(a1 - alph1)*x[index_i][findex] + y2*(a2 - alph2)*x[index_j][findex];
	}

	//TODO: update error cache using new lagrange mults

	// update the alpha array with the new values
	alpha[index_i] = a1;
	alpha[index_j] = a2;

	return 1;
}// solver::update

}
;// namespace

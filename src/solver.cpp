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

	// Instantiate a string cache with at most (x) elements.
	*cache = new double_cache_t(N/2);

	/*********** TEST IMPL OF CACHE ***********************/
//	// Insert data into the cache.
//	std::string quote_1 = "Number is the within of all things. -Pythagoras";
//	cache->insert(4, quote_1);
//
//	// Fetch it out.
//	std::clog << cache->fetch(4) << std::endl;
	/*******************************************************/
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
	double s = y1*y2;

	double alpha1old = alpha[index_i];
	double alpha2old = alpha[index_j];
	double alpha1updated = 0;
	double alpha2updated = 0;

	double E2 = cache->fetch(index_j);
	double E1 = cache->fetch(index_i);

	// compute L and H via equations
	double H = 0;
	double L = 0;
	if (s < 0) {
		L = getMax(0, (alpha2old-alpha1old));
		H = getMin(C, (C+alpha2old-alpha1old));
	} else {
		L = getMax(0, (alpha2old+alpha1old-C));
		H = getMin(C, (alpha2old+alpha1old));
	}

	if (L == H) {
		return 0;
	}

	double k11 = kernel(x[index_i], x[index_i]);//<x1,x1>;
	double k12 = kernel(x[index_i], x[index_j]);//<x1,x2>;
	double k22 = kernel(x[index_j], x[index_j]);//<x2,x2>;
	double eta = k11 + k22 - (2 * k12);

	if (eta > 0) {
		alpha2updated = alpha2old + (y2 * (E1-E2) / eta);
		if (alpha2updated < L) {
			alpha2updated = L;
		} else if (alpha2updated > H) {
			alpha2updated = H;
		}

	} else {
		// calculate these objectives
		double aa2 = L;
		double aa1 = alpha1old + s * (alpha2old-aa2);
		double Lobj = aa1 + aa2; // + (y2 * L * x[]) - b: objective function at a2 = L;
		for (int elementIndex = 0; elementIndex < N; elementIndex++) {
			Lobj += ((-y1*aa1/2) * y[elementIndex] * kernel(x[elementIndex], x[index_i])) +
					((-y2*aa2/2) * y[elementIndex] * kernel(x[elementIndex], x[index_j]));
		}

		aa2 = H;
		aa1 = alpha1old + s * (alpha2old-aa2);
		double Hobj = aa1 + aa2; // + (y2 * H * x[]) - b: objective function at a2 = H;
		for (int elementIndex = 0; elementIndex < N; elementIndex++) {
			Hobj += ((-y1*aa1/2) * y[elementIndex] * kernel(x[elementIndex], x[index_i])) +
					((-y2*aa2/2) * y[elementIndex] * kernel(x[elementIndex], x[index_j]));
		}

		if (Lobj < (Hobj - EPS)) {
			alpha2updated = L;
		} else if (Lobj > (Hobj + EPS)) {
			alpha2updated = H;
		} else {
			alpha2updated = alpha2old;
		}
	}

	if (abs(alpha2updated-alpha2old) < EPS*(alpha2updated + alpha2old + EPS)) {
		return 0;
	}

	alpha1updated = alpha1old + s*(alpha2old-alpha2updated);

	// update bias (threshold) to reflect change in alphas
	// 2.3 Computing the Threshold
	double b1 = E1 + y1*((alpha1updated - alpha1old) * k11) + y2*((alpha2updated - alpha2old) * k12) + b;
	double b2 = E2 + y1*((alpha1updated - alpha1old) * k12) + y2*((alpha2updated - alpha2old) * k22) + b;
	if (((alpha1updated == H) | (alpha1updated == L)) & ((alpha2updated == H) | (alpha2updated == L))) { // could this be simplified?
		b = (b1 + b2) / 2;
	} else {
		// alphas not at bounds: b1 should be equal to b2
		b = b1;
	}

	// update weight vector
	// 2.4 An Optimization for Linear SVMs
	for (int findex = 0; findex < M; findex++) {
		w[findex] = w[findex] + y1*(alpha1updated - alpha1old)*x[index_i][findex] + y2*(alpha2updated - alpha2old)*x[index_j][findex];
	}

	//TODO: update error cache using new lagrange mults
	//SMO.error = SMO.error + w1*K(:,i1) + w2*K(:,i2) + bold - SMO.bias; // NEED TO UPDATE EVERY ENTRY INPLACE
	// SMO.error(i1) = 0.0; SMO.error(i2) = 0.0;
	cache->insert(index_i, 0.0);
	cache->insert(index_j, 0.0);

	// update the alpha array with the new values
	alpha[index_i] = alpha1updated;
	alpha[index_j] = alpha2updated;

	return 1;
}// solver::update

}
;// namespace

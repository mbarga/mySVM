#include <mysvm.h>
#include <solver.h>

#define getMax(a,b) a>b?a:b
#define getMin(a,b) a<b?a:b

namespace MySVM
{

// Solver class constructor
Solver::Solver()
{
	// variables initialized in main();
}

double Solver::kernel(double* x[], int index_i, int index_j)
{
	double dotProduct = 0;

	for (int i = 0; i < features; i++)
	{
		double val1 = x[index_i][i];
		double val2 = x[index_j][i];
		dotProduct += (val1 * val2);
	}

	return dotProduct;
}

int Solver::examine(int index_j)
{
	double y2 = y[index_j];
	double alph2 = alpha[index_j];
	double E2 = error[index_j];
	double r2 = E2 * y2;

	int index_i = 0;
	std::vector<int>::iterator iter;
	std::vector<int> nonBoundAlphaIdx;

	if (((r2 < -EPS) && (alph2 < C)) || ((r2 > EPS) && (alph2 > 0)))
	{
		//find number and indices of non-zero and non-C alphas
		for (index_i = 0; index_i < length; index_i++)
		{
			if (alpha[index_i] < 0)
			{
				std::clog << "alpha returned was < 0" << std::endl;
			}
			if ((alpha[index_i] > EPS)
					&& ((alpha[index_i] < (C - EPS))
							|| (alpha[index_i] > (C + EPS))))
			{
				// push non-bound alpha index into vector cache
				nonBoundAlphaIdx.push_back(index_i);
			}
		}

		if (nonBoundAlphaIdx.size() > 1)
		{
			// perform second choice heuristic to choose index_i
			index_i = 0;
			// choose multiplier to maximize the step taken; i.e. max(|E1 - E2|);
			double errortemp = 0;
			for (iter = nonBoundAlphaIdx.begin();
					iter != nonBoundAlphaIdx.end(); ++iter)
			{
				if (abs(error[*iter] - E2) > errortemp)
				{
					index_i = *iter;
				}
				errortemp = abs(error[*iter] - E2);
			}
			//TODO: if () the errortemp doesnt stay 0?
			if (update(index_i, index_j))
			{
				return 1;
			}
		}

		// else loop over all non-zero and non-c alpha, starting at a random point
		//TODO: start at random point- is this right?
		random_shuffle(nonBoundAlphaIdx.begin(), nonBoundAlphaIdx.end());
		for (iter = nonBoundAlphaIdx.begin(); iter != nonBoundAlphaIdx.end();
				++iter)
		{
			index_i = *iter;
			if (update(index_i, index_j))
			{
				return 1;
			}
		}

		// else loop over all possible i1, starting at random point
		//TODO: start at random point
		//int updated = 0;
		for (index_i = 0; index_i < length; index_i++)
		{
			if (update(index_i, index_j) == 1)
			{
				return 1;
			}
		}
	} // if error > tolerance

	return 0;
}

int Solver::update(int index_i, int index_j)
{
	if (index_i == index_j)
	{
		return 0;
	}

	double y1 = y[index_i];
	double y2 = y[index_j];
	double s = y1 * y2;

	double alpha1old = alpha[index_i];
	double alpha2old = alpha[index_j];
	double alpha1updated = 0;
	double alpha2updated = 0;

	double E2 = error[index_j];
	double E1 = error[index_i];

	// compute L and H via equations
	double H = 0;
	double L = 0;
	if (s < 0)
	{
		L = getMax(0, (alpha2old-alpha1old));
		H = getMin(C, (C+alpha2old-alpha1old));
	}
	else
	{
		L = getMax(0, (alpha2old+alpha1old-C));
		H = getMin(C, (alpha2old+alpha1old));
	}

	if (L == H)
	{
		return 0;
	}

	double k11 = kernel(x, index_i, index_i); //<x1,x1>;
	double k12 = kernel(x, index_i, index_j); //<x1,x2>;
	double k22 = kernel(x, index_j, index_j); //<x2,x2>;
	double eta = k11 + k22 - 2 * k12;

	if (eta > 0)
	{
		alpha2updated = alpha2old + y2 * (E1 - E2) / eta;
		if (alpha2updated < L)
		{
			alpha2updated = L;
		}
		else if (alpha2updated > H)
		{
			alpha2updated = H;
		}

	}
	else
	{
		//NOTE: this is a rare case, but SVM should work regardless
		std::clog << "DEBUG:: eta was negative" << std::endl;

		// calculate these objectives
		double aa2 = L;
		double aa1 = alpha1old + s * (alpha2old - aa2);
		double Lobj = aa1 + aa2; // + (y2 * L * x[]) - b: objective function at a2 = L;
		for (int elementIndex = 0; elementIndex < length; elementIndex++)
		{
			Lobj += ((-y1 * aa1 / 2) * y[elementIndex]
					* kernel(x, elementIndex, index_i))
					+ ((-y2 * aa2 / 2) * y[elementIndex]
							* kernel(x, elementIndex, index_j));
		}

		aa2 = H;
		aa1 = alpha1old + s * (alpha2old - aa2);
		double Hobj = aa1 + aa2; // + (y2 * H * x[]) - b: objective function at a2 = H;
		for (int elementIndex = 0; elementIndex < length; elementIndex++)
		{
			Hobj += ((-y1 * aa1 / 2) * y[elementIndex]
					* kernel(x, elementIndex, index_i))
					+ ((-y2 * aa2 / 2) * y[elementIndex]
							* kernel(x, elementIndex, index_j));
		}

		if (Lobj < (Hobj - EPS))
		{
			alpha2updated = L;
		}
		else if (Lobj > (Hobj + EPS))
		{
			alpha2updated = H;
		}
		else
		{
			alpha2updated = alpha2old;
		}
	}

	//take care of numerical errors
	//TODO: debug
	//printf("new %f; old %f\n",alpha2updated,alpha2old);
	if (alpha2updated < EPS) {
		alpha2updated = 0;
	} else if (alpha2updated > (C - EPS)) {
		alpha2updated = C;
	}

	double diff = fabs(alpha2updated - alpha2old);
	double thresh = EPS * (alpha2updated+alpha2old+EPS);
	//TODO: debug
	//printf("a1new:%f, a1old:%f | diff:%f ? thresh:%f ",alpha2updated,alpha2old,diff,thresh);
	if (diff < thresh)
	{
		std::cout << "DEBUG:: alpha unchanged" << std::endl;
		return 0;
	}

	// update alpha_1
	alpha1updated = alpha1old + s * (alpha2old - alpha2updated);
	if (alpha1updated < 0)
	{
		alpha1updated = 0;
	}
	else if (alpha1updated > C)
	{
		alpha1updated = C;
	}

	// update bias (threshold) to reflect change in alphas
	// 2.3 Computing the Threshold
	double bold = b;
	double deltaalpha1 = alpha1updated - alpha1old;
	double deltaalpha2 = alpha2updated - alpha2old;

	double b1 = E1 + y1 * deltaalpha1 * k11 + y2 * deltaalpha2 * k12 + b;
	double b2 = E2 + y1 * deltaalpha1 * k12 + y2 * deltaalpha2 * k22 + b;

	if (!((alpha1updated == H) | (alpha1updated == L)))
	{
		b = b1;
	}
	else if (!((alpha2updated == H) | (alpha2updated == L)))
	{
		b = b2;
	}
	else
	{
		b = (b1 + b2) / 2;
	}

	// update weight vector
	// 2.4 An Optimization for Linear SVMs
	//TODO: look at this closer
	for (int findex = 0; findex < features; findex++)
	{
		w[findex] = w[findex] + y1 * deltaalpha1 * x[index_i][findex]
				+ y2 * deltaalpha2 * x[index_j][findex];
	}

	// update error cache using new lagrange mults
	for (int i; i < length; i++)
	{
		error[i] += y1 * deltaalpha1 * kernel(x, i, index_i)
				+ y2 * deltaalpha2 * kernel(x, i, index_j) - b + bold;
	}
	//NOTE: maybe unnecessary: set the errors to exactly 0 for the optimized alphas
	error[index_i] = 0.0;
	error[index_j] = 0.0;

	// update the alpha array with the new values
	alpha[index_i] = alpha1updated;
	alpha[index_j] = alpha2updated;

	return 1;
}

}
;
// namespace

#include "mysvm.h"
#include "solver.h"
#include "log.h"

// NOTICE: dont include name space in main()

/**
 *	This is the main entry point for the svm training algorithm. Implements Platt's SMO for C-SVM
 * 
 */

int main(int argc, char **argv) {

	int index = 0;
	int numChanged = 0;
	bool examineAll = true;

	std::clog.rdbuf(new Log("foo", LOG_LOCAL0));

	MySVM::Solver solver;

	//TODO: Get the data from file

	while ((numChanged > 0) || (examineAll)) {

		numChanged = 0;

		// OUTER LOOP (first lagrange multiplier)
		// first, loop over entire training set 
		if (examineAll) {
			for (index = 0; index < N; index++) {
				numChanged += solver.examine(index);
			}

			// else iterate over multipliers that are not at the bounds
		} else {
			for (index = 0; index < N; index++) {
				if ((abs(solver.alpha[index]) > EPS) && ((abs(
						solver.alpha[index] < (C - EPS))) || (abs(
						solver.alpha[index]) > (C + EPS)))) {
					numChanged += solver.examine(index);
				}
			}
		}

		// if subset was unchanged, loop over entire set again
		if (examineAll) {
			examineAll = false;
		} else if (numChanged == 0) {
			examineAll = true;
		}

	}// while

	/*********** TEST IMPL OF CACHE ***********************/
	std::clog << kLogNotice << "test log message" << std::endl;
	std::clog << "the default is debug level" << std::endl;
	/*******************************************************/
	/*********** TEST IMPL OF CACHE ***********************/
	// Typedef our template for easy of readability and use.
	typedef LRUCache<int,std::string> string_cache_t;

	// Instantiate a string cache with at most three elements.
	string_cache_t *cache = new string_cache_t(3);

	// Insert data into the cache.
	std::string quote_1 = "Number is the within of all things. -Pythagoras";
	cache->insert( 4, quote_1 );

	// Fetch it out.
	std::clog << cache->fetch( 4 ) << std::endl;
	/*******************************************************/

	printf("EXITING\n");
	return 0;
} // main

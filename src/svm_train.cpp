#include "mysvm.h"
#include "solver.h"
#include "log.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

// NOTICE: dont include name space in main()

static char *line = NULL;
static int max_line_len;
int read_problem(const char *filename);
static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

MySVM::Solver solver;
double* x_space;

/**
 *	This is the main entry point for the svm training algorithm. Implements Platt's SMO for C-SVM
 * 
 */

int main(int argc, char **argv) {

	int index = 0;
	int numChanged = 0;
	bool examineAll = true;

	// instantiate logging
	std::clog.rdbuf(new Log("mysvm_log", LOG_LOCAL0));

	/*********** TEST IMPL OF LOG ***********************/
	std::clog << kLogNotice << "Log initialized..." << std::endl;
	std::clog << "the default is debug level" << std::endl;
	/*******************************************************/

	char input_file_name[1024] = "/home/mbarga/Workbench/git/mySVM/src/test.input";
	int status = read_problem(input_file_name);

	if (status != 0) {
		std::clog << "failed to properly read in input, aborting" << std::endl;
		return 1;
	}
	//TODO: clean up the info that was read in
	for (int i = 0; i < 10; i++) {
		std::clog << solver.y[i] << std::endl;
	}

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

	printf("EXITING\n");

	// w = alpha * y
	// b = bias
	return 0;

} // main

// read in a problem (in svmlight format)
int read_problem(const char *filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
	int length = 0;

	if(fp == NULL) {
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL) {
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++length;
	}
	rewind(fp);

	solver.y = Malloc(double, length);
	solver.x = Malloc(double *, length);
	x_space = Malloc(double, elements);

	max_index = 0;
	j=0;
	for(i=0; i<length; i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		solver.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			return 1;

		solver.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			return 1;

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			//x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || *endptr != '\0' ) //|| x_space[j].index <= inst_max_index)
				return 1;
			else
				inst_max_index = (int) strtol(idx,&endptr,10); //x_space[j].index;

			x_space[j] = strtod(val,&endptr);
			if(endptr == val || (*endptr != '\0' && !isspace(*endptr)))
				return 1;

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		//x_space[j++].index = -1;
	}

	fclose(fp);
	return 0;
}

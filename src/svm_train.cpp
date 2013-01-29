#include "mysvm.h"
#include "solver.h"
#include "log.h"

// NOTICE: dont include name space in main()

static char *line = NULL;
static int max_line_len;

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

/**
 *	This is the main entry point for the svm training algorithm. Implements Platt's SMO for C-SVM
 * 
 */

int main(int argc, char **argv) {

	int index = 0;
	int numChanged = 0;
	bool examineAll = true;

	char input_file_name[1024];

	// get the data from file
	// int status = read_problem(input_file_name);

	// instantiate logging
	std::clog.rdbuf(new Log("foo", LOG_LOCAL0));

	/*********** TEST IMPL OF LOG ***********************/
	std::clog << kLogNotice << "test log message" << std::endl;
	std::clog << "the default is debug level" << std::endl;
	/*******************************************************/

	MySVM::Solver solver;

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

	//inserting a comment to change later
} // main

// read in a problem (in svmlight format)
int read_problem(const char *filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
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
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,N);
	prob.x = Malloc(struct svm_node *,N);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			return 1;

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			return 1;

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				return 1;
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				return 1;

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	fclose(fp);
	return 0;
}

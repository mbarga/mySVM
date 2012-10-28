CXX ?= g++ 
#CXX ?= clang 
CFLAGS = -Wall -O3
 
all: svm_train

# -g tells it to add support for debugger
svm_train: 
	$(CXX) $(CFLAGS) -g ./src/solver.cpp ./src/svm_train.cpp -o model -lm

clean:
	rm -f *~ svm.o model 

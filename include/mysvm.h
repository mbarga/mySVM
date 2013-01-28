/** @file mysvm.h
 * @brief General data I/O functions, etc.
 */
#ifndef _MYSVM_H
#define _MYSVM_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept> 	// std::runtime_error
#include <string.h>
namespace MySVM {
// NOTE: The file writer implements a C++ idiom, "RAII" for encapsulated resource management

class file {
public:
	file(const char* filename) :
		file_(std::fopen(filename, "w+")) {
		if (!file_) {
			throw std::runtime_error("file open failure");
		}
	}

	~file() {
		if (std::fclose(file_)) {
			// failed to flush latest changes.
			// handle it
		}
	}

	void write(const char* str) {
		if (EOF == std::fputs(str, file_)) {
			throw std::runtime_error("file write failure");
		}
	}

private:
	std::FILE* file_;

	// prevent copying and assignment; not implemented
	file(const file &);
	file& operator=(const file &);
};
}
; // namespace
#endif

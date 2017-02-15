#ifndef SRC_UTIL_HB2TACO_H_
#define SRC_UTIL_HB2TACO_H_

#include <fstream>

namespace hb2taco {

  void readFile(std::ifstream &hbfile, int** colptr, int** rowind, double** values);
  void writeFile();

  void readHeader(std::ifstream &hbfile,
		  std::string* title, std::string* key,
		  int* totcrd, int* ptrcrd, int* indcrd, int* valcrd, int* rhscrd,
		  std::string* mxtype, int* nrow, int* ncol, int* nnzero, int*neltvl,
		  std::string* ptrfmt, std::string* indfmt, std::string* valfmt, std::string* rhsfmt);
  void writeHeader();
  void readIndices(std::ifstream &hbfile, int linesize, int indices[]);
  void writePointers();
  void writeRowIndices();
  void readValues(std::ifstream &hbfile, int linesize, double values[]);
  void writeValues();
  // Useless for Taco
  void readRHS();
  void writeRHS();
}

#endif /* SRC_UTIL_HB2TACO_H_ */

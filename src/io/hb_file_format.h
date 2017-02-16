#ifndef SRC_UTIL_HB2TACO_H_
#define SRC_UTIL_HB2TACO_H_

#include <fstream>

namespace taco {
namespace io {
namespace hb {
void readFile(std::ifstream &hbfile,
              int* nrow, int* ncol,
              int** colptr, int** rowind, double** values);
void writeFile(std::ofstream &hbfile, std::string key,
               int nrow, int ncol, int nnzero,
               int ptrsize, int indsize, int valsize,
               int* colptr, int* rowind, double* values);

void readHeader(std::ifstream &hbfile,
                std::string* title, std::string* key,
                int* totcrd, int* ptrcrd, int* indcrd, int* valcrd, int* rhscrd,
                std::string* mxtype, int* nrow, int* ncol, int* nnzero, int* neltvl,
                std::string* ptrfmt, std::string* indfmt, std::string* valfmt, std::string* rhsfmt);
void writeHeader(std::ofstream &hbfile,
                 std::string title, std::string key,
                 int totcrd, int ptrcrd, int indcrd, int valcrd, int rhscrd,
                 std::string mxtype, int nrow, int ncol, int nnzero, int neltvl,
                 std::string ptrfmt, std::string indfmt, std::string valfmt, std::string rhsfmt);
void readIndices(std::ifstream &hbfile, int linesize, int indices[]);
void writeIndices(std::ofstream &hbfile, int indsize, int linesize, int indices[]);
void readValues(std::ifstream &hbfile, int linesize, double values[]);
void writeValues(std::ofstream &hbfile, int valuesize, int valperline, double values[]);
// Useless for Taco
void readRHS();
void writeRHS();
}}}
#endif /* SRC_UTIL_HB2TACO_H_ */

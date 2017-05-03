#ifndef IO_RB_FILE_FORMAT_H
#define IO_RB_FILE_FORMAT_H

#include <istream>
#include <ostream>
#include <string>

namespace taco {
class TensorBase;
namespace io {
namespace rb {

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
                std::string* mxtype, int* nrow,
                int* ncol, int* nnzero, int* neltvl,
                std::string* ptrfmt, std::string* indfmt,
                std::string* valfmt, std::string* rhsfmt);
void writeHeader(std::ofstream &hbfile,
                 std::string title, std::string key,
                 int totcrd, int ptrcrd, int indcrd, int valcrd, int rhscrd,
                 std::string mxtype, int nrow, int ncol, int nnzero, int neltvl,
                 std::string ptrfmt, std::string indfmt,
                 std::string valfmt, std::string rhsfmt);
void readIndices(std::ifstream &hbfile, int linesize, int indices[]);
void writeIndices(std::ofstream &hbfile, int indsize,
                  int linesize, int indices[]);
void readValues(std::ifstream &hbfile, int linesize, double values[]);
void writeValues(std::ofstream &hbfile, int valuesize,
                 int valperline, double values[]);
// Useless for Taco
void readRHS();
void writeRHS();

/// Read an hb matrix from a file.
TensorBase read(std::string filename);

/// Read an hb matrix from a stream
TensorBase read(std::istream& stream);

/// Write an hb matrix to a file
void write(std::string filename, const TensorBase& tensor);

/// Write an hb matrix to a stream
void write(std::ostream& stream, const TensorBase& tensor);

}}}
#endif

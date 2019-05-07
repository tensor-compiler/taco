/// Read and write the HB Harwell-Boeing Sparse File Format

#ifndef TACO_FILE_IO_RB_H
#define TACO_FILE_IO_RB_H

#include <istream>
#include <ostream>
#include <string>

#include "taco/format.h"

namespace taco {
class TensorBase;
class Format;

void readFile(std::istream &hbfile,
              int* nrow, int* ncol,
              int** colptr, int** rowind, double** values);
void writeFile(std::ostream &hbfile, std::string key,
               int nrow, int ncol, int nnzero,
               int ptrsize, int indsize, int valsize,
               int* colptr, int* rowind, double* values);

void readHeader(std::istream &hbfile,
                std::string* title, std::string* key,
                int* totcrd, int* ptrcrd, int* indcrd, int* valcrd, int* rhscrd,
                std::string* mxtype, int* nrow,
                int* ncol, int* nnzero, int* neltvl,
                std::string* ptrfmt, std::string* indfmt,
                std::string* valfmt, std::string* rhsfmt);
void writeHeader(std::ostream &hbfile,
                 std::string title, std::string key,
                 int totcrd, int ptrcrd, int indcrd, int valcrd, int rhscrd,
                 std::string mxtype, int nrow, int ncol, int nnzero, int neltvl,
                 std::string ptrfmt, std::string indfmt,
                 std::string valfmt, std::string rhsfmt);
void readIndices(std::istream &hbfile, int linesize, int indices[]);
void writeIndices(std::ostream &hbfile, int indsize,
                  int linesize, int indices[]);
void readValues(std::istream &hbfile, int linesize, double values[]);
void writeValues(std::ostream &hbfile, int valuesize,
                 int valperline, double values[]);
// Useless for Taco
void readRHS();
void writeRHS();

/// Read an rb matrix from a file.
TensorBase readRB(std::string filename, const ModeFormat& modetype, 
                  bool pack=true);

/// Read an rb matrix from a file.
TensorBase readRB(std::string filename, const Format& format, bool pack=true);

/// Read an rb matrix from a stream
TensorBase readRB(std::istream& stream, const ModeFormat& modetype, 
                  bool pack=true);

/// Read an rb matrix from a stream
TensorBase readRB(std::istream& stream, const Format& format, bool pack=true);

/// Write an rb matrix to a file
void writeRB(std::string filename, const TensorBase& tensor);

/// Write an rb matrix to a stream
void writeRB(std::ostream& stream, const TensorBase& tensor);

}
#endif

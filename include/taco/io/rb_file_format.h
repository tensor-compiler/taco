/// Read and write the HB Harwell-Boeing Sparse File Format

#ifndef IO_RB_FILE_FORMAT_H
#define IO_RB_FILE_FORMAT_H

#include <istream>
#include <ostream>
#include <string>

namespace taco {
class TensorBase;
class Format;
namespace io {
namespace rb {

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

/// Read an hb matrix from a file.
TensorBase read(std::string filename, const Format& format, bool pack = true);

/// Read an hb matrix from a stream
TensorBase read(std::istream& stream, const Format& format, bool pack = true);

/// Write an hb matrix to a file
void write(std::string filename, const TensorBase& tensor);

/// Write an hb matrix to a stream
void write(std::ostream& stream, const TensorBase& tensor);

}}}
#endif

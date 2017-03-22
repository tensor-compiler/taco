/*
 * benchmark.h
 */

#ifndef INCLUDE_TACO_UTIL_BENCHMARK_H_
#define INCLUDE_TACO_UTIL_BENCHMARK_H_

#include <time.h>
#include <algorithm>

using namespace std;

namespace taco {
namespace util {

struct timeResults {
  double mean;
  double stdev;
  double median;
  int size;
  friend std::ostream& operator<<(std::ostream &os, const timeResults &t) {
    return os << "[[ taco Time Results: " << endl
              << "    ** mean:        " << t.mean << endl
              << "    ** deviation:   " << t.stdev << endl
              << "    ** median:      " << t.median << endl
              << "    ** sample size: " << t.size << "  ]]" << endl;
  }
};

class Benchmark {
public:
  Benchmark(int repeat) : repeat(repeat) {
    init();
  };
  ~Benchmark() {};

  void init() {
    begins.resize(repeat, 0.0);
    ends.resize(repeat, 0.0);
  }

  void start(int i) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    begins[i] = 1000*ts.tv_sec + 1e-6 * ts.tv_nsec;
  }
  void stop(int i) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    ends[i] = 1000*ts.tv_sec + 1e-6 * ts.tv_nsec;
  }

  // Compute mean, standard deviation and median
  timeResults getResults() {
    timeResults result;
    vector<double> times(repeat);
    double mean=0.0;
    // times = ends - begins
    transform (ends.begin(), ends.end(),
               begins.begin(), times.begin(), minus<double>());
    sort(times.begin(), times.end());
    // remove 10% worst and best cases
    mean = accumulate(times.begin()+(int)(repeat*0.1),
                      times.end()-(int)(repeat*0.1), 0.0);
    int size = repeat - (int)(repeat*0.2);
    result.size = size;
    mean = mean/size;
    result.mean = mean;

    vector<double> diff(size);
    transform(times.begin()+(int)(repeat*0.1),
              times.end()-(int)(repeat*0.1),
              diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = inner_product(diff.begin(), diff.end(),
                                  diff.begin(), 0.0);
    result.stdev = sqrt(sq_sum / size);

    if (size%2)
      result.median = times[size/2];
    else
      result.median = (times[size/2-1] + times[size/2]) / 2;
    return result;
  }

protected:
  int repeat;
  vector<double> begins;
  vector<double> ends;
  timespec ts;

};
}}

#define TACO_BENCHMARK(CODE,REPEAT,RES) { \
    taco::util::Benchmark Bench(REPEAT); \
    for(int i=0; i<REPEAT; i++){ \
      Bench.start(i); \
      CODE; \
      Bench.stop(i); \
    } \
    RES = Bench.getResults(); \
  }

#endif /* INCLUDE_TACO_UTIL_BENCHMARK_H_ */

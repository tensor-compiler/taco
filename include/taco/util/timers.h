#ifndef TACO_UTIL_BENCHMARK_H
#define TACO_UTIL_BENCHMARK_H

#include <time.h>
#include <algorithm>
#include <numeric>
#include <cmath>


using namespace std;

namespace taco {
namespace util {

struct TimeResults {
  double mean;
  double stdev;
  double median;
  int size;

  friend std::ostream& operator<<(std::ostream& os, const TimeResults& t) {
    if (t.size == 1) {
      return os << t.mean;
    }
    else {
      return os << "  mean:   " << t.mean   << endl
                << "  stdev:  " << t.stdev  << endl
                << "  median: " << t.median;
    }
  }
};

/// Monotonic timer that can be called multiple times and that computes
/// statistics such as mean and median from the calls.
class Timer {
public:
  Timer() {}

  void start() {
    begin = timespec();
    end = timespec();
    clock_gettime(CLOCK_MONOTONIC, &begin);
  }

  void stop() {
    clock_gettime(CLOCK_MONOTONIC, &end);
    begins.push_back(1000*begin.tv_sec + 1e-6 * begin.tv_nsec);
    ends.push_back(1000*end.tv_sec + 1e-6 * end.tv_nsec);
  }

  // Compute mean, standard deviation and median
  TimeResults getResult() {
    int repeat = begins.size();

    TimeResults result;
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
    result.median = (size % 2)
                    ? result.median = times[size/2]
                    : result.median = (times[size/2-1] + times[size/2]) / 2;
    return result;
  }

protected:
  vector<double> begins;
  vector<double> ends;

  timespec begin;
  timespec end;
};
}}

#define TACO_TIME_REPEAT(CODE, REPEAT, RES) {  \
    taco::util::Timer timer;                   \
    for(int i=0; i<REPEAT; i++) {              \
      timer.start();                           \
      CODE;                                    \
      timer.stop();                            \
    }                                          \
    RES = timer.getResult();                  \
  }

#endif

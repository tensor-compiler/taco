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

static inline double toMilliseconds(const timespec& ts) {
  return 1000*ts.tv_sec + 1e-6 * ts.tv_nsec;
}

/// Monotonic timer that can be called multiple times and that computes
/// statistics such as mean and median from the calls.
class Timer {
public:
  void start() {
    begin = timespec();
    end = timespec();
    clock_gettime(CLOCK_MONOTONIC, &begin);
  }

  void stop() {
    clock_gettime(CLOCK_MONOTONIC, &end);
    begins.push_back(toMilliseconds(begin));
    ends.push_back(toMilliseconds(end));
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
                    ? times[size/2]
                    : (times[size/2-1] + times[size/2]) / 2;
    return result;
  }

protected:
  vector<double> begins;
  vector<double> ends;

  timespec begin;
  timespec end;
};


/// Monotonic timer that prints results when stopped.
class PrintTimer {
public:
  PrintTimer(string timerName = "") : timerGroup(true), isTiming(false) {
    if (timerName != "") {
      std::cout << timerName << std::endl;
    }
  }

  void start(string name) {
    this->timingName = name;
    taco_iassert(!isTiming) << "Called PrintTimer::start twice in a row";
    isTiming = true;
    clock_gettime(CLOCK_MONOTONIC, &begin);
  }

  void stop() {
    clock_gettime(CLOCK_MONOTONIC, &end);
    taco_iassert(isTiming)
        << "Called PrintTimer::stop without first calling start";
    if (timerGroup) {
      std::cout << "  ";
    }
    std::cout << timingName << ": "
              << (toMilliseconds(end)-toMilliseconds(begin)) << " ms"
              << std::endl;
    isTiming = false;
  }

private:
  bool timerGroup;
  string timingName;
  timespec begin;
  timespec end;
  bool isTiming;
};


/// Monotonic scoped print timer.
class ScopedTimer {
public:
  ScopedTimer(string name) {
    timer.start(name);
  }

  ~ScopedTimer() {
    timer.stop();
  }

private:
  PrintTimer timer;
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

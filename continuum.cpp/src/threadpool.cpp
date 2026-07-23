/*
 * threadpool.cpp - Thread pool with ARM big-core pinning.
 *
 * Phase B: Mobile thread management.
 * - Detects big cores via /sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq
 * - Pins threads via sched_setaffinity (Linux/Android only)
 * - Work-stealing parallel_for for matmul parallelization
 */

#include "threadpool.h"

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#include <fstream>
#endif

#include <algorithm>
#include <cstring>

namespace continuum {

// ============================================================================
// ARM big core detection
// ============================================================================
int ThreadPool::detect_big_cores() {
#ifdef __linux__
    // Read max frequency of each CPU core. Big cores have higher frequency.
    // Typical: little cores 1.4-1.8 GHz, big cores 2.0-3.0 GHz
    std::vector<int64_t> freqs;
    for (int i = 0; i < 16; i++) {  // max 16 cores
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        std::ifstream f(path);
        if (f.is_open()) {
            int64_t freq;
            f >> freq;
            freqs.push_back(freq);
            f.close();
        } else {
            break;  // no more cores
        }
    }

    if (freqs.empty()) return 2;  // fallback

    // Find max frequency → big cores are those within 80% of max
    int64_t max_freq = *std::max_element(freqs.begin(), freqs.end());
    int big_count = 0;
    for (int64_t f : freqs) {
        if (f >= max_freq * 0.8) big_count++;
    }

    // Cap at 4 threads — more threads cause cache thrashing for 100M model
    return std::min(big_count, 4);
#else
    return 2;  // non-Linux fallback
#endif
}

void ThreadPool::pin_to_big_cores() {
#ifdef __linux__
    int big_count = detect_big_cores();
    if (big_count <= 0) return;

    // Build CPU set: use cores with highest frequency
    // Strategy: use the LAST N cores (big cores are usually higher-numbered)
    int total_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = total_cores - big_count; i < total_cores; i++) {
        if (i >= 0) CPU_SET(i, &cpuset);
    }
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#endif
}

// ============================================================================
// ThreadPool implementation
// ============================================================================
ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = (size_t)detect_big_cores();
        if (num_threads == 0) num_threads = 2;
    }
    threads_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; i++) {
        threads_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::set_num_threads(size_t n) {
    // Stop existing threads
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
    stop_ = false;

    // Start new threads
    threads_.reserve(n);
    for (size_t i = 0; i < n; i++) {
        threads_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this]() {
        return tasks_.empty() && active_tasks_ == 0;
    });
}

void ThreadPool::parallel_for(size_t n, std::function<void(size_t, size_t)> func) {
    size_t nthreads = threads_.size();
    if (nthreads <= 1 || n < 256) {
        // Single-threaded fast path for small workloads
        func(0, n);
        return;
    }

    // Split [0, n) into nthreads chunks
    size_t chunk = (n + nthreads - 1) / nthreads;
    int enqueued = 0;

    for (size_t start = 0; start < n; start += chunk) {
        size_t end = std::min(start + chunk, n);
        active_tasks_++;
        enqueue([func, start, end]() {
            func(start, end);
        });
        enqueued++;
    }

    wait_all();
}

void ThreadPool::worker_loop() {
    // Pin each worker to big cores
    pin_to_big_cores();

    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return stop_ || !tasks_.empty();
            });

            if (stop_ && tasks_.empty()) return;

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        task();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_tasks_--;
            if (tasks_.empty() && active_tasks_ == 0) {
                done_cv_.notify_one();
            }
        }
    }
}

// Global thread pool (lazy init)
ThreadPool& global_thread_pool() {
    static ThreadPool pool(0);  // auto-detect big cores
    return pool;
}

} // namespace continuum

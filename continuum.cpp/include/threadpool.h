/*
 * threadpool.h — Thread pool with ARM big-core pinning for Continuum SLM.
 *
 * Phase B: Thread management for mobile inference.
 * - Pins worker threads to "big" cores via sched_setaffinity
 * - Work-stealing task queue with minimal synchronization
 * - parallel_for helper for splitting matmul work across threads
 * - 2-4 threads optimal for 100M param model on phone
 */

#ifndef CONTINUUM_THREADPOOL_H
#define CONTINUUM_THREADPOOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <cstdint>

namespace continuum {

class ThreadPool {
public:
    // num_threads=0 → auto-detect big cores (default: min(4, big_core_count))
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    // Enqueue a task (returns immediately, task runs on worker thread)
    void enqueue(std::function<void()> task);

    // Wait for all enqueued tasks to complete
    void wait_all();

    // Parallel for: split [0, n) into chunks, run func(start, end) on each chunk
    // Blocks until all chunks complete.
    void parallel_for(size_t n, std::function<void(size_t, size_t)> func);

    // Number of worker threads
    size_t num_threads() const { return threads_.size(); }

    // Pin current thread to big cores (call from main thread too)
    static void pin_to_big_cores();

    // Detect number of big cores on ARM big.LITTLE
    static int detect_big_cores();

    // Set thread count (must be called before any parallel_for)
    void set_num_threads(size_t n);

private:
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;       // workers wait for tasks
    std::condition_variable done_cv_;  // main thread waits for completion
    std::atomic<int> active_tasks_{0};
    std::atomic<bool> stop_{false};

    void worker_loop();
};

// Global thread pool (lazy-initialized, auto-pinned to big cores)
ThreadPool& global_thread_pool();

} // namespace continuum

#endif // CONTINUUM_THREADPOOL_H

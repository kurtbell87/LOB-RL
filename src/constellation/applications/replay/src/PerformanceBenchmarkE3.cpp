/**
 * @file PerformanceBenchmarkE3.cpp
 * @brief Phase E3: Performance Benchmarking for Batch Aggregator.
 *
 * This tool reads one or multiple .dbn / .dbn.zst files concurrently,
 * each using an independent BatchAggregator. It measures ingestion
 * throughput (messages per second), aggregator timings, and
 * concurrency scaling.
 *
 * Usage Example:
 *   ./replay_perf_e3 /path/to/dbn_folder [num_threads=4] [batch_size=50000]
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "replay/BatchAggregator.hpp"

namespace fs = std::filesystem;
using namespace constellation::applications::replay;

/**
 * @brief Holds aggregator stats plus timing results for each worker thread.
 */
struct ThreadResult {
  BatchAggregatorStats stats;
  double wall_seconds{0.0};
};

static ThreadResult ProcessFilesThread(const std::vector<std::string>& files,
                                       std::uint32_t batch_size,
                                       int idx)
{
  ThreadResult result;
  try {
    BatchAggregator aggregator;
    BatchAggregatorConfig cfg;
    cfg.batch_size = batch_size;
    cfg.enable_logging = false; // disable aggregator logs for raw speed
    aggregator.Initialize(cfg);

    auto t_start = std::chrono::steady_clock::now();

    for (const auto& path : files) {
      std::cerr << "[Worker " << idx << "] Processing file: " << path << std::endl;
      aggregator.ProcessSingleFile(path);
    }

    auto t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> dur = (t_end - t_start);
    result.wall_seconds = dur.count();

    result.stats = aggregator.GetStats();
  } catch (const std::exception& ex) {
    std::cerr << "[Worker " << idx
              << "] EXCEPTION: " << ex.what() << std::endl;
  } catch (...) {
    std::cerr << "[Worker " << idx
              << "] Unknown exception" << std::endl;
  }
  return result;
}

static std::vector<std::vector<std::string>>
DistributeFiles(const std::vector<std::string>& all_files,
                std::size_t num_threads)
{
  std::vector<std::vector<std::string>> result(num_threads);
  for (std::size_t i = 0; i < all_files.size(); ++i) {
    std::size_t t = i % num_threads;
    result[t].push_back(all_files[i]);
  }
  return result;
}

static std::vector<std::string> GatherDbnFiles(const std::string& path) {
  std::vector<std::string> files;
  fs::path p(path);
  if (fs::is_directory(p)) {
    for (const auto& entry : fs::directory_iterator(p)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      auto ext = entry.path().extension().string();
      if (ext == ".dbn" || ext == ".zst" || ext == ".dbn.zst") {
        files.push_back(entry.path().string());
      }
    }
    std::sort(files.begin(), files.end());
  } else if (fs::is_regular_file(p)) {
    // Single file
    files.push_back(path);
  }
  return files;
}

// Prints usage and available test data
void printUsage(char* program_name) {
  std::cerr << "Usage:\n  " << program_name
            << " <dbn_path_or_folder> [num_threads=4] [batch_size=50000]\n\n";
            
  // Try to detect and list test data
  fs::path current_path = fs::current_path();
  fs::path test_data_path = current_path;
  
  // Search for test_data in common locations
  std::vector<fs::path> possible_paths = {
    current_path / "test_data",
    current_path / "cpp" / "test_data",
    current_path.parent_path() / "test_data",
    current_path.parent_path() / "cpp" / "test_data",
    current_path.parent_path().parent_path() / "test_data",
    current_path.parent_path().parent_path() / "cpp" / "test_data"
  };

  for (const auto& path : possible_paths) {
    if (fs::exists(path) && fs::is_directory(path)) {
      test_data_path = path;
      std::cerr << "Detected test data at: " << test_data_path.string() << "\n\n";
      std::cerr << "Available test files:\n";
      for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file()) {
          auto ext = entry.path().extension().string();
          if (ext == ".dbn" || ext == ".zst" || ext == ".dbn.zst") {
            std::cerr << "  " << entry.path().filename().string() << "\n";
          }
        }
      }
      std::cerr << "\nExample:\n  " << program_name << " " 
                << test_data_path.string() << " 4 50000\n";
      break;
    }
  }
}

// Check if arg is a test framework command
bool isTestFrameworkCommand(const std::string& arg) {
  return arg.substr(0, 2) == "--" || 
         arg == "-h" || 
         arg == "-v" || 
         arg == "-l";
}

int main(int argc, char** argv)
{
  // Special handling for test framework args
  if (argc > 1 && isTestFrameworkCommand(argv[1])) {
    std::cerr << "Detected test framework command: " << argv[1] << "\n"
              << "This program requires proper benchmark parameters.\n";
    printUsage(argv[0]);
    return 0;
  }

  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string path = argv[1];
  int num_threads = 4; // Default value
  
  // Safely parse num_threads with error handling
  if (argc > 2) {
    try {
      std::string arg = argv[2];
      if (!isTestFrameworkCommand(arg) && !arg.empty()) {
        num_threads = std::stoi(arg);
      }
    } catch (const std::invalid_argument& e) {
      std::cerr << "Warning: Invalid number of threads specified. Using default (4)." << std::endl;
    } catch (const std::out_of_range& e) {
      std::cerr << "Warning: Thread count out of range. Using default (4)." << std::endl;
    }
  }
  
  std::uint32_t batch_size = 50000; // Default value
  
  // Safely parse batch_size with error handling
  if (argc > 3) {
    try {
      std::string arg = argv[3];
      if (!isTestFrameworkCommand(arg) && !arg.empty()) {
        batch_size = static_cast<std::uint32_t>(std::stoul(arg));
      }
    } catch (const std::invalid_argument& e) {
      std::cerr << "Warning: Invalid batch size specified. Using default (50000)." << std::endl;
    } catch (const std::out_of_range& e) {
      std::cerr << "Warning: Batch size out of range. Using default (50000)." << std::endl;
    }
  }

  // Gather .dbn or .zst files
  auto all_files = GatherDbnFiles(path);
  if (all_files.empty()) {
    std::cerr << "[WARNING] No .dbn or .zst files found at: " << path << std::endl;
    
    // Try a default test data location if path might be a test framework arg
    if (isTestFrameworkCommand(path)) {
      fs::path current_path = fs::current_path();
      fs::path test_data_path = current_path / "cpp" / "test_data";
      
      if (fs::exists(test_data_path) && fs::is_directory(test_data_path)) {
        std::cerr << "[INFO] Trying default test data path: " << test_data_path.string() << std::endl;
        all_files = GatherDbnFiles(test_data_path.string());
      }
    }
    
    if (all_files.empty()) {
      printUsage(argv[0]);
      return 0;
    }
  }
  
  std::cerr << "[INFO] Found " << all_files.size() << " DBN files\n";

  // Distribute among threads
  if (num_threads < 1) {
    std::cerr << "[WARNING] Invalid thread count (" << num_threads 
              << "), setting to 1" << std::endl;
    num_threads = 1;
  }
  auto subsets = DistributeFiles(all_files, num_threads);

  std::cerr << "[INFO] Launching " << num_threads
            << " worker thread(s), Batch aggregator batch_size=" << batch_size << std::endl;

  // Start timer
  auto t0 = std::chrono::steady_clock::now();

  // Launch threads
  std::vector<std::thread> threads;
  std::vector<ThreadResult> results(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&results, &subsets, i, batch_size]() {
      results[i] = ProcessFilesThread(subsets[i], batch_size, i);
    });
  }

  // Join
  for (auto& th : threads) {
    th.join();
  }

  // End timer
  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> total_dur = (t1 - t0);
  double total_wall_s = total_dur.count();

  // Combine aggregator stats
  std::uint64_t total_records = 0ULL;
  std::uint64_t total_mbo     = 0ULL;
  std::uint64_t total_micro   = 0ULL;

  double sum_worker_wall = 0.0;

  for (int i = 0; i < num_threads; ++i) {
    total_records += results[i].stats.total_records.load();
    total_mbo     += results[i].stats.total_mbo_messages.load();
    total_micro   += results[i].stats.total_microseconds.load();
    sum_worker_wall += results[i].wall_seconds;
  }

  // Summaries
  double msg_per_sec = 0.0;
  if (total_wall_s > 0.0) {
    msg_per_sec = static_cast<double>(total_records) / total_wall_s;
  }
  double aggregator_s = static_cast<double>(total_micro) / 1e6; // aggregator's internal measure

  std::cout << "\n===== Batch Aggregator E3 Performance Benchmark =====\n";
  std::cout << "Threads           : " << num_threads << "\n";
  std::cout << "Batch size        : " << batch_size << "\n";
  std::cout << "Total DBN files   : " << all_files.size() << "\n";
  std::cout << "--------------------------------------------------\n";
  std::cout << "Total records     : " << total_records << "\n";
  std::cout << "Total MBO msgs    : " << total_mbo << "\n";
  std::cout << "Aggregator CPU time (s) : " << aggregator_s << "\n";
  std::cout << "Avg aggregator msg/s    : "
            << (aggregator_s > 0.0
                 ? (static_cast<double>(total_records) / aggregator_s)
                 : 0.0)
            << "\n";
  std::cout << "--------------------------------------------------\n";
  std::cout << "Total real wall time (s) : " << total_wall_s << "\n";
  std::cout << "Overall throughput (msg/s): " << msg_per_sec << "\n";
  std::cout << "(Sum of per-thread wall)  : " << sum_worker_wall << "\n";
  std::cout << "==================================================\n";

  return 0;
}

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/resource.h>
#include <unistd.h>

#include <csignal>

#include <atomic>
#include <chrono>
#include <set>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "storage_backend.h"

namespace mooncake {
namespace fs = std::filesystem;

class BucketReclaimTest : public ::testing::Test {
   protected:
    void SetUp() override {
        data_path_ = (fs::current_path() / "bucket_reclaim_test_data").string();
        fs::remove_all(data_path_);
        fs::create_directories(data_path_);
        config_.storage_filepath = data_path_;
        backend_ = std::make_unique<BucketStorageBackend>(
            config_, BucketBackendConfig{});
        ASSERT_TRUE(backend_->Init());
    }

    void TearDown() override {
        backend_.reset();
        fs::remove_all(data_path_);
    }

    int64_t Offload(const std::unordered_map<std::string, std::string>& kvs) {
        std::unordered_map<std::string, std::vector<Slice>> batch;
        for (const auto& [key, value] : kvs) {
            batch.emplace(
                key, std::vector<Slice>{
                         Slice{const_cast<char*>(value.data()), value.size()}});
        }
        auto result = backend_->BatchOffload(
            batch,
            [](const std::vector<std::string>&,
               std::vector<StorageObjectMetadata>&) { return ErrorCode::OK; });
        EXPECT_TRUE(result.has_value());
        return result.value_or(-1);
    }

    std::optional<std::string> Load(const std::string& key, size_t size) {
        std::string buf(size, '\0');
        std::unordered_map<std::string, Slice> slices;
        slices.emplace(key, Slice{buf.data(), size});
        if (!backend_->BatchLoad(slices)) {
            return std::nullopt;
        }
        return buf;
    }

    std::string DataFile(int64_t bucket_id) const {
        return data_path_ + "/" + std::to_string(bucket_id) + ".bucket";
    }

    // Open descriptors of this process that still point at a deleted file
    // under the data directory: their blocks stay allocated.
    int DeletedOpenFiles() const {
        int count = 0;
        for (const auto& entry : fs::directory_iterator("/proc/self/fd")) {
            std::error_code ec;
            auto target = fs::read_symlink(entry.path(), ec).string();
            if (!ec && target.rfind(data_path_, 0) == 0 &&
                target.find("(deleted)") != std::string::npos) {
                ++count;
            }
        }
        return count;
    }

    std::string data_path_;
    FileStorageConfig config_;
    std::unique_ptr<BucketStorageBackend> backend_;
};

TEST_F(BucketReclaimTest, AllKeysDeadDeletesBucket) {
    const std::string value(4096, 'a');
    const int64_t bucket_id =
        Offload({{"k1", value}, {"k2", value}, {"k3", value}});
    ASSERT_EQ(Load("k1", value.size()), value);

    EXPECT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k1", "k2", "k3"}}}), 1u);
    EXPECT_FALSE(fs::exists(DataFile(bucket_id)));
    EXPECT_EQ(DeletedOpenFiles(), 0);
    for (const auto* key : {"k1", "k2", "k3"}) {
        EXPECT_FALSE(backend_->IsExist(key).value());
    }
    int64_t cursor = -1;
    EXPECT_TRUE(backend_->GetReclaimCandidates(-1, 16, cursor).empty());
}

TEST_F(BucketReclaimTest, PartiallyDeadBucketStays) {
    const std::string v1(4096, 'a'), v2(4096, 'b'), v3(4096, 'c');
    const int64_t bucket_id = Offload({{"k1", v1}, {"k2", v2}, {"k3", v3}});

    EXPECT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k1"}}}), 0u);
    EXPECT_TRUE(fs::exists(DataFile(bucket_id)));
    EXPECT_FALSE(backend_->IsExist("k1").value());
    EXPECT_FALSE(Load("k1", v1.size()).has_value());
    EXPECT_EQ(Load("k2", v2.size()), v2);
    EXPECT_EQ(Load("k3", v3.size()), v3);

    EXPECT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k2", "k3"}}}), 1u);
    EXPECT_FALSE(fs::exists(DataFile(bucket_id)));
}

TEST_F(BucketReclaimTest, KeyReofloadedIntoNewerBucketKeepsMapping) {
    const std::string old_value(4096, 'o'), new_value(4096, 'n');
    const int64_t old_bucket = Offload({{"k1", old_value}, {"k2", old_value}});
    ASSERT_EQ(backend_->RemoveDeadKeys({{old_bucket, {"k1"}}}), 0u);
    const int64_t new_bucket = Offload({{"k1", new_value}});
    ASSERT_NE(new_bucket, old_bucket);

    // A verdict taken against the old bucket must not touch the re-offload.
    EXPECT_EQ(backend_->RemoveDeadKeys({{old_bucket, {"k1", "k2"}}}), 1u);
    EXPECT_FALSE(fs::exists(DataFile(old_bucket)));
    EXPECT_TRUE(fs::exists(DataFile(new_bucket)));
    EXPECT_EQ(Load("k1", new_value.size()), new_value);
}

TEST_F(BucketReclaimTest, DeadBytesStayAccountedUntilBucketDeleted) {
    const std::string value(4096, 'a');
    const int64_t bucket_id = Offload({{"k1", value}, {"k2", value}});
    const int64_t full = backend_->GetStoreMetadata()->total_size;
    ASSERT_GT(full, 0);
    ASSERT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k1"}}}), 0u);
    EXPECT_EQ(backend_->GetStoreMetadata()->total_size, full);
    ASSERT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k2"}}}), 1u);
    EXPECT_EQ(backend_->GetStoreMetadata()->total_size, 0);
}

TEST_F(BucketReclaimTest, ScanAndRestartSeeOnlyLiveMappings) {
    const std::string old_value(4096, 'o'), new_value(4096, 'n');
    const int64_t old_bucket = Offload({{"k1", old_value}, {"k2", old_value}});
    ASSERT_EQ(backend_->RemoveDeadKeys({{old_bucket, {"k1"}}}), 0u);

    std::vector<std::string> keys;
    std::vector<StorageObjectMetadata> metadatas;
    std::vector<int64_t> buckets;
    ASSERT_TRUE(backend_->BucketScan(0, keys, metadatas, buckets, 1000));
    EXPECT_EQ(keys, std::vector<std::string>{"k2"});

    const int64_t new_bucket = Offload({{"k1", new_value}});
    ASSERT_GT(new_bucket, old_bucket);
    // Restart: both buckets list k1; the newest one must win.
    backend_.reset();
    backend_ =
        std::make_unique<BucketStorageBackend>(config_, BucketBackendConfig{});
    ASSERT_TRUE(backend_->Init());
    EXPECT_EQ(Load("k1", new_value.size()), new_value);
    EXPECT_EQ(Load("k2", old_value.size()), old_value);
}

TEST_F(BucketReclaimTest, DroppedKeysStayDroppedAfterRestart) {
    const std::string value(4096, 'a');
    const int64_t bucket_id = Offload({{"k1", value}, {"k2", value}});
    ASSERT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k1"}}}), 0u);
    const int64_t total = backend_->GetStoreMetadata()->total_size;

    backend_.reset();
    backend_ =
        std::make_unique<BucketStorageBackend>(config_, BucketBackendConfig{});
    ASSERT_TRUE(backend_->Init());
    EXPECT_FALSE(backend_->IsExist("k1").value());
    EXPECT_EQ(Load("k2", value.size()), value);
    EXPECT_EQ(backend_->GetStoreMetadata()->total_size, total);
    std::vector<std::string> keys;
    std::vector<StorageObjectMetadata> metadatas;
    std::vector<int64_t> buckets;
    ASSERT_TRUE(backend_->BucketScan(0, keys, metadatas, buckets, 1));
    EXPECT_EQ(keys, std::vector<std::string>{"k2"});

    EXPECT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k2"}}}), 1u);
    EXPECT_EQ(backend_->GetStoreMetadata()->total_size, 0);
    size_t files = 0;
    for (const auto& entry : fs::directory_iterator(data_path_)) {
        files += entry.path().stem() == std::to_string(bucket_id);
    }
    EXPECT_EQ(files, 0u);
}

TEST_F(BucketReclaimTest, TornTombstoneTailKeepsLaterDrops) {
    const std::string value(4096, 'a');
    const int64_t bucket_id =
        Offload({{"k1", value}, {"k2", value}, {"k3", value}});
    ASSERT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k1"}}}), 0u);
    {
        // A crash in the middle of the next append.
        std::ofstream out(
            data_path_ + "/" + std::to_string(bucket_id) + ".dead",
            std::ios::binary | std::ios::app);
        const uint32_t size = 64;
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out.write("torn", 4);
    }
    auto restart = [&] {
        backend_.reset();
        backend_ = std::make_unique<BucketStorageBackend>(
            config_, BucketBackendConfig{});
        ASSERT_TRUE(backend_->Init());
    };
    restart();
    EXPECT_FALSE(backend_->IsExist("k1").value());
    ASSERT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k2"}}}), 0u);
    restart();
    EXPECT_FALSE(backend_->IsExist("k1").value());
    EXPECT_FALSE(backend_->IsExist("k2").value());
    EXPECT_EQ(Load("k3", value.size()), value);
}

TEST_F(BucketReclaimTest, UndeletableBucketStaysForRetry) {
    const std::string value(4096, 'a');
    const int64_t bucket_id = Offload({{"k1", value}});
    const std::string base = data_path_ + "/" + std::to_string(bucket_id);
    // Neither the metadata nor a tombstone can be written off.
    fs::rename(base + ".meta", base + ".meta.saved");
    for (const auto* suffix : {".meta", ".dead"}) {
        fs::create_directories(base + suffix + "/x");
    }
    EXPECT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k1"}}}), 0u);
    EXPECT_TRUE(backend_->IsExist("k1").value());
    EXPECT_EQ(Load("k1", value.size()), value);

    for (const auto* suffix : {".meta", ".dead"}) {
        fs::remove_all(base + suffix);
    }
    fs::rename(base + ".meta.saved", base + ".meta");
    EXPECT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k1"}}}), 1u);
    EXPECT_FALSE(backend_->IsExist("k1").value());
    EXPECT_FALSE(fs::exists(DataFile(bucket_id)));
}

TEST_F(BucketReclaimTest, FailedTombstoneAppendRollsBack) {
    const std::string value(4096, 'a');
    const int64_t bucket_id =
        Offload({{"k1", value}, {"k2", value}, {"k3", value}});
    ASSERT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k1"}}}), 0u);
    const std::string tombstone =
        data_path_ + "/" + std::to_string(bucket_id) + ".dead";
    const auto size = fs::file_size(tombstone);
    {
        // The next append stops after 3 bytes, like a full disk.
        auto* old_handler = std::signal(SIGXFSZ, SIG_IGN);
        rlimit old_limit{};
        ASSERT_EQ(0, getrlimit(RLIMIT_FSIZE, &old_limit));
        rlimit limit = old_limit;
        limit.rlim_cur = size + 3;
        ASSERT_EQ(0, setrlimit(RLIMIT_FSIZE, &limit));
        EXPECT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k2"}}}), 0u);
        ASSERT_EQ(0, setrlimit(RLIMIT_FSIZE, &old_limit));
        std::signal(SIGXFSZ, old_handler);
    }
    EXPECT_TRUE(backend_->IsExist("k2").value());
    EXPECT_EQ(fs::file_size(tombstone), size);
    ASSERT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k2"}}}), 0u);

    backend_.reset();
    backend_ =
        std::make_unique<BucketStorageBackend>(config_, BucketBackendConfig{});
    ASSERT_TRUE(backend_->Init());
    EXPECT_FALSE(backend_->IsExist("k1").value());
    EXPECT_FALSE(backend_->IsExist("k2").value());
    EXPECT_EQ(Load("k3", value.size()), value);
}

TEST_F(BucketReclaimTest, KeptMetadataIsDeletedAtRestart) {
    const std::string value(4096, 'a');
    const int64_t bucket_id = Offload({{"k1", value}, {"k2", value}});
    const std::string base = data_path_ + "/" + std::to_string(bucket_id);
    // The metadata cannot be removed, the tombstone can.
    fs::rename(base + ".meta", base + ".meta.saved");
    fs::create_directories(base + ".meta/x");
    EXPECT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k1", "k2"}}}), 0u);
    EXPECT_FALSE(backend_->IsExist("k1").value());
    EXPECT_TRUE(fs::exists(DataFile(bucket_id)));
    fs::remove_all(base + ".meta");
    fs::rename(base + ".meta.saved", base + ".meta");

    backend_.reset();
    backend_ =
        std::make_unique<BucketStorageBackend>(config_, BucketBackendConfig{});
    ASSERT_TRUE(backend_->Init());
    EXPECT_FALSE(backend_->IsExist("k1").value());
    EXPECT_FALSE(backend_->IsExist("k2").value());
    size_t files = 0;
    for (const auto& entry : fs::directory_iterator(data_path_)) {
        files += entry.path().stem() == std::to_string(bucket_id);
    }
    EXPECT_EQ(files, 0u);
}

TEST_F(BucketReclaimTest, ConcurrentReadsDuringDeleteCompleteOrMiss) {
    const std::string value(1 << 20, 'r');
    const int64_t bucket_id = Offload({{"k1", value}});
    std::atomic<bool> stop{false};
    std::atomic<int> hits{0}, misses{0}, corrupt{0};
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&] {
            while (!stop.load()) {
                auto loaded = Load("k1", value.size());
                if (!loaded) {
                    misses.fetch_add(1);
                } else if (*loaded == value) {
                    hits.fetch_add(1);
                } else {
                    corrupt.fetch_add(1);
                }
            }
        });
    }
    while (hits.load() < 8) {
        std::this_thread::yield();
    }
    EXPECT_EQ(backend_->RemoveDeadKeys({{bucket_id, {"k1"}}}), 1u);
    while (misses.load() < 8) {
        std::this_thread::yield();
    }
    stop.store(true);
    for (auto& reader : readers) {
        reader.join();
    }
    EXPECT_EQ(corrupt.load(), 0);
    EXPECT_FALSE(fs::exists(DataFile(bucket_id)));
    EXPECT_EQ(DeletedOpenFiles(), 0);
}

TEST_F(BucketReclaimTest, CandidatesWrapAroundFromCursor) {
    const std::string value(4096, 'a');
    const int64_t b1 = Offload({{"k1", value}});
    const int64_t b2 = Offload({{"k2", value}});
    const int64_t b3 = Offload({{"k3", value}});

    int64_t cursor = -1;
    auto first = backend_->GetReclaimCandidates(cursor, 2, cursor);
    ASSERT_EQ(first.size(), 2u);
    EXPECT_EQ(first[0].first, b1);
    EXPECT_EQ(first[1].first, b2);
    EXPECT_EQ(first[1].second, std::vector<std::string>{"k2"});
    auto second = backend_->GetReclaimCandidates(cursor, 2, cursor);
    ASSERT_EQ(second.size(), 2u);
    EXPECT_EQ(second[0].first, b3);
    EXPECT_EQ(second[1].first, b1);
}

// Synthetic 10k-bucket index: a lap with the default interval/cycle budget
// takes cycle/interval passes regardless of the bucket count. Disabled by
// default (writes 10k bucket files); run with --gtest_also_run_disabled_tests.
TEST_F(BucketReclaimTest, DISABLED_FullLapPassesAt10kBuckets) {
    constexpr int kBuckets = 10'000, kKeysPerBucket = 500;
    const std::string value(64, 'v');
    const auto write_start = std::chrono::steady_clock::now();
    for (int b = 0; b < kBuckets; ++b) {
        std::unordered_map<std::string, std::string> kvs;
        for (int k = 0; k < kKeysPerBucket; ++k) {
            kvs.emplace("aux/" + std::to_string(b) + "/" + std::to_string(k),
                        value);
        }
        Offload(kvs);
    }
    LOG(INFO) << "RECLAIM_BENCH holder wrote " << kBuckets << " buckets in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - write_start)
                     .count()
              << " s";

    const FileStorageConfig defaults;
    const size_t total_keys = backend_->GetStoreMetadata()->total_keys;
    ASSERT_EQ(total_keys, size_t{kBuckets} * kKeysPerBucket);
    const size_t max_keys = total_keys * defaults.reclaim_interval_seconds /
                            defaults.reclaim_cycle_seconds;
    std::set<int64_t> seen;
    int64_t cursor = -1;
    int passes = 0;
    double max_pass_ms = 0;
    while (seen.size() < kBuckets) {
        const auto t0 = std::chrono::steady_clock::now();
        auto candidates =
            backend_->GetReclaimCandidates(cursor, max_keys, cursor);
        max_pass_ms =
            std::max(max_pass_ms, std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count());
        for (const auto& [bucket_id, keys] : candidates) {
            seen.insert(bucket_id);
        }
        ++passes;
    }
    LOG(INFO) << "RECLAIM_BENCH holder buckets=" << kBuckets
              << " keys=" << total_keys << " keys_per_pass=" << max_keys
              << " passes_per_lap=" << passes
              << " lap_s=" << passes * defaults.reclaim_interval_seconds
              << " max_candidates_ms=" << max_pass_ms;
    EXPECT_LE(
        passes * defaults.reclaim_interval_seconds,
        defaults.reclaim_cycle_seconds + defaults.reclaim_interval_seconds);
}

}  // namespace mooncake

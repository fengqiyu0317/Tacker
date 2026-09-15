#pragma once

#include "runtime/Kernel.h"

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace tacker {
namespace runtime {

class KernelRegistry {
public:
    void registerKernel(const std::shared_ptr<KernelSpec>& spec);
    std::shared_ptr<const KernelSpec> findKernel(const std::string& key) const;
    std::vector<std::string> kernelKeys() const;

    void registerMixed(const MixedKernelRegistration& registration);
    bool findMixed(const std::string& left_key, const std::string& right_key,
                   MixedKernelRegistration* result) const;
    std::vector<MixedKernelRegistration> findMixedCandidates(
        const std::string& left_key, const std::string& right_key) const;

private:
    typedef std::pair<std::string, std::string> PairKey;
    mutable std::mutex mutex_;
    std::map<std::string, std::shared_ptr<KernelSpec> > kernels_;
    std::map<PairKey, std::vector<MixedKernelRegistration> > mixed_;
};

struct ProfileRecord {
    ProfileRecord()
        : enabled(false), valid(false), solo_sum_us(0.0), mixed_p50_us(0.0),
          mixed_p95_us(0.0), primary_slowdown_percent(0.0),
          median_throughput_fps(0.0), selected_for_deployment(false) {}

    bool enabled;
    bool valid;
    double solo_sum_us;
    double mixed_p50_us;
    double mixed_p95_us;
    double primary_slowdown_percent;
    // Whole-sequence throughput is the selection objective. The leaf and
    // primary latency fields above are retained as diagnostics only.
    double median_throughput_fps;
    bool selected_for_deployment;
    std::string manifest_hash;

    double estimatedSavingsUs() const { return solo_sum_us - mixed_p50_us; }
};

class ProfileRegistry {
public:
    void set(const std::string& left_key, const std::string& right_key,
             const std::string& mixed_key, const ProfileRecord& profile);
    bool find(const std::string& left_key, const std::string& right_key,
              const std::string& mixed_key, ProfileRecord* result) const;
    void disable(const std::string& left_key, const std::string& right_key,
                 const std::string& mixed_key);

private:
    struct ProfileKey {
        std::string left;
        std::string right;
        std::string mixed;
        bool operator<(const ProfileKey& other) const;
    };

    mutable std::mutex mutex_;
    std::map<ProfileKey, ProfileRecord> profiles_;
};

}  // namespace runtime
}  // namespace tacker

#include "runtime/Registry.h"

#include <cmath>

namespace tacker {
namespace runtime {

void KernelRegistry::registerKernel(const std::shared_ptr<KernelSpec>& spec) {
    if (!spec || spec->key.empty()) throw std::invalid_argument("kernel key cannot be empty");
    if (spec->block.x == 0 || spec->block.y == 0 || spec->block.z == 0) {
        throw std::invalid_argument("kernel block dimensions must be non-zero");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (kernels_.count(spec->key) != 0) {
        throw std::invalid_argument("duplicate kernel key: " + spec->key);
    }
    kernels_[spec->key] = spec;
}

std::shared_ptr<const KernelSpec> KernelRegistry::findKernel(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, std::shared_ptr<KernelSpec> >::const_iterator it = kernels_.find(key);
    return it == kernels_.end() ? std::shared_ptr<const KernelSpec>() : it->second;
}

std::vector<std::string> KernelRegistry::kernelKeys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    for (std::map<std::string, std::shared_ptr<KernelSpec> >::const_iterator it = kernels_.begin();
         it != kernels_.end(); ++it) {
        result.push_back(it->first);
    }
    return result;
}

void KernelRegistry::registerMixed(const MixedKernelRegistration& registration) {
    if (!registration.mixed_spec) throw std::invalid_argument("mixed kernel spec is required");
    if (registration.mixed_spec->key.empty()) {
        throw std::invalid_argument("mixed kernel key cannot be empty");
    }
    if (registration.mixed_spec->block.x == 0 || registration.mixed_spec->block.y == 0 ||
        registration.mixed_spec->block.z == 0) {
        throw std::invalid_argument("mixed kernel block dimensions must be non-zero");
    }
    if (registration.left_key.empty() || registration.right_key.empty()) {
        throw std::invalid_argument("mixed kernel input keys cannot be empty");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, std::shared_ptr<KernelSpec> >::const_iterator left =
        kernels_.find(registration.left_key);
    std::map<std::string, std::shared_ptr<KernelSpec> >::const_iterator right =
        kernels_.find(registration.right_key);
    if (left == kernels_.end() || right == kernels_.end()) {
        throw std::invalid_argument("mixed kernel inputs must already be registered");
    }
    if (!left->second->mixable || !right->second->mixable) {
        throw std::invalid_argument("mixed kernel inputs must be marked mixable");
    }
    const PairKey key(registration.left_key, registration.right_key);
    std::vector<MixedKernelRegistration>& candidates = mixed_[key];
    for (std::vector<MixedKernelRegistration>::const_iterator it = candidates.begin();
         it != candidates.end(); ++it) {
        if (it->mixed_spec->key == registration.mixed_spec->key) {
            throw std::invalid_argument("duplicate mixed kernel registration");
        }
    }
    candidates.push_back(registration);
}

bool KernelRegistry::findMixed(const std::string& left_key, const std::string& right_key,
                               MixedKernelRegistration* result) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const PairKey key(left_key, right_key);
    std::map<PairKey, std::vector<MixedKernelRegistration> >::const_iterator it = mixed_.find(key);
    if (it == mixed_.end()) return false;
    if (result) *result = it->second.front();
    return true;
}

std::vector<MixedKernelRegistration> KernelRegistry::findMixedCandidates(
    const std::string& left_key, const std::string& right_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const PairKey key(left_key, right_key);
    std::map<PairKey, std::vector<MixedKernelRegistration> >::const_iterator it =
        mixed_.find(key);
    return it == mixed_.end() ? std::vector<MixedKernelRegistration>() : it->second;
}

bool ProfileRegistry::ProfileKey::operator<(const ProfileKey& other) const {
    if (left != other.left) return left < other.left;
    if (right != other.right) return right < other.right;
    return mixed < other.mixed;
}

void ProfileRegistry::set(const std::string& left_key, const std::string& right_key,
                          const std::string& mixed_key, const ProfileRecord& profile) {
    if (left_key.empty() || right_key.empty() || mixed_key.empty()) {
        throw std::invalid_argument("profile keys cannot be empty");
    }
    if (profile.selected_for_deployment &&
        (!profile.enabled || !profile.valid ||
         !std::isfinite(profile.median_throughput_fps) ||
         profile.median_throughput_fps <= 0.0 || profile.manifest_hash.empty())) {
        throw std::invalid_argument(
            "selected deployment profile requires enabled, valid, finite whole-run FPS and manifest hash");
    }
    ProfileKey key;
    key.left = left_key;
    key.right = right_key;
    key.mixed = mixed_key;
    std::lock_guard<std::mutex> lock(mutex_);
    profiles_[key] = profile;
}

bool ProfileRegistry::find(const std::string& left_key, const std::string& right_key,
                           const std::string& mixed_key, ProfileRecord* result) const {
    ProfileKey key;
    key.left = left_key;
    key.right = right_key;
    key.mixed = mixed_key;
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<ProfileKey, ProfileRecord>::const_iterator it = profiles_.find(key);
    if (it == profiles_.end()) return false;
    if (result) *result = it->second;
    return true;
}

void ProfileRegistry::disable(const std::string& left_key, const std::string& right_key,
                              const std::string& mixed_key) {
    ProfileKey key;
    key.left = left_key;
    key.right = right_key;
    key.mixed = mixed_key;
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<ProfileKey, ProfileRecord>::iterator it = profiles_.find(key);
    if (it != profiles_.end()) it->second.enabled = false;
}

}  // namespace runtime
}  // namespace tacker

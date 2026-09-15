#pragma once

#include "runtime/Types.h"

#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tacker {
namespace runtime {

enum class KernelLaunchApi {
    Runtime,
    Driver
};

struct KernelSpec {
    KernelSpec()
        : native_handle(NULL), static_shared_memory(0), dynamic_shared_memory(0),
          registers_per_thread(0), mixable(false), launch_api(KernelLaunchApi::Runtime) {}

    std::string key;
    std::string symbol;
    std::string architecture;
    std::string abi_hash;
    void* native_handle;
    Dim3 grid;
    Dim3 block;
    std::size_t static_shared_memory;
    std::size_t dynamic_shared_memory;
    unsigned int registers_per_thread;
    bool mixable;
    KernelLaunchApi launch_api;
};

// Owns the bytes used by a CUDA-style void** argument vector. Pointer-valued
// arguments are copied as pointer values; an optional lifetime token can keep
// the allocation that they reference alive until the invocation is released.
class OwnedArgumentPack {
public:
    OwnedArgumentPack() {}

    template <typename T>
    void addScalar(const T& value) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "kernel scalar arguments must be trivially copyable");
        addBytes(&value, sizeof(T), std::shared_ptr<void>());
    }

    void addPointer(void* value, const std::shared_ptr<void>& lifetime = std::shared_ptr<void>()) {
        addBytes(&value, sizeof(value), lifetime);
    }

    void addConstPointer(const void* value,
                         const std::shared_ptr<void>& lifetime = std::shared_ptr<void>()) {
        const void* copied = value;
        addBytes(&copied, sizeof(copied), lifetime);
    }

    void addRaw(const void* value, std::size_t size,
                const std::shared_ptr<void>& lifetime = std::shared_ptr<void>()) {
        if (size != 0 && value == NULL) {
            throw std::invalid_argument("non-empty kernel argument cannot have a null source");
        }
        addBytes(value, size, lifetime);
    }

    void append(const OwnedArgumentPack& other);
    std::size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }
    std::vector<void*> argumentPointers() const;

private:
    struct Entry {
        std::shared_ptr<std::vector<unsigned char> > bytes;
        std::shared_ptr<void> lifetime;
    };

    void addBytes(const void* value, std::size_t size, const std::shared_ptr<void>& lifetime);
    std::vector<Entry> entries_;
};

struct KernelInvocation {
    KernelInvocation() : stream(0), use_launch_override(false),
                         dynamic_shared_memory_override(0) {}

    explicit KernelInvocation(const std::shared_ptr<const KernelSpec>& spec_)
        : spec(spec_), stream(0), use_launch_override(false),
          dynamic_shared_memory_override(0) {}

    Dim3 launchGrid() const;
    Dim3 launchBlock() const;
    std::size_t dynamicSharedMemory() const;

    std::shared_ptr<const KernelSpec> spec;
    OwnedArgumentPack arguments;
    StreamHandle stream;
    bool use_launch_override;
    Dim3 grid_override;
    Dim3 block_override;
    std::size_t dynamic_shared_memory_override;
    std::shared_ptr<void> lifetime;
};

typedef std::function<OwnedArgumentPack(const KernelInvocation&, const KernelInvocation&)>
    MixedArgumentComposer;

struct MixedKernelRegistration {
    std::string left_key;
    std::string right_key;
    std::shared_ptr<const KernelSpec> mixed_spec;
    MixedArgumentComposer compose_arguments;
};

}  // namespace runtime
}  // namespace tacker

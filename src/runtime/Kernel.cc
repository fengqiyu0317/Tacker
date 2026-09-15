#include "runtime/Kernel.h"

namespace tacker {
namespace runtime {

void OwnedArgumentPack::addBytes(const void* value, std::size_t size,
                                 const std::shared_ptr<void>& lifetime) {
    Entry entry;
    entry.bytes.reset(new std::vector<unsigned char>(size));
    if (size != 0) {
        std::memcpy(entry.bytes->data(), value, size);
    }
    entry.lifetime = lifetime;
    entries_.push_back(entry);
}

void OwnedArgumentPack::append(const OwnedArgumentPack& other) {
    entries_.insert(entries_.end(), other.entries_.begin(), other.entries_.end());
}

std::vector<void*> OwnedArgumentPack::argumentPointers() const {
    std::vector<void*> result;
    result.reserve(entries_.size());
    for (std::vector<Entry>::const_iterator it = entries_.begin(); it != entries_.end(); ++it) {
        result.push_back(it->bytes->empty() ? NULL : static_cast<void*>(it->bytes->data()));
    }
    return result;
}

Dim3 KernelInvocation::launchGrid() const {
    if (use_launch_override) return grid_override;
    if (!spec) throw std::logic_error("kernel invocation has no specification");
    return spec->grid;
}

Dim3 KernelInvocation::launchBlock() const {
    if (use_launch_override) return block_override;
    if (!spec) throw std::logic_error("kernel invocation has no specification");
    return spec->block;
}

std::size_t KernelInvocation::dynamicSharedMemory() const {
    if (use_launch_override) return dynamic_shared_memory_override;
    if (!spec) throw std::logic_error("kernel invocation has no specification");
    return spec->dynamic_shared_memory;
}

}  // namespace runtime
}  // namespace tacker

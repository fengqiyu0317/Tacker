#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tacker {
namespace runtime {

typedef std::uint64_t NodeId;
typedef std::uint64_t SubmissionId;
typedef std::uint64_t StreamHandle;

struct EventHandle {
    EventHandle() : value(0) {}
    explicit EventHandle(std::uint64_t value_) : value(value_) {}

    bool valid() const { return value != 0; }
    bool operator==(const EventHandle& other) const { return value == other.value; }
    bool operator!=(const EventHandle& other) const { return !(*this == other); }

    std::uint64_t value;
};

struct Dim3 {
    Dim3() : x(1), y(1), z(1) {}
    Dim3(unsigned int x_, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}

    unsigned int x;
    unsigned int y;
    unsigned int z;
};

enum class NodeType {
    GPTB,
    Solo,
    Opaque,
    ExternalEvent,
    HostFence
};

enum class NodeState {
    Pending,
    Ready,
    Submitted,
    Completed
};

enum class SubmissionRole {
    LatencyCritical,
    BestEffort
};

struct DeviceProperties {
    DeviceProperties()
        : device_ordinal(-1), major(0), minor(0), multiprocessor_count(0),
          max_threads_per_block(0), shared_memory_per_block(0) {}

    int device_ordinal;
    int major;
    int minor;
    int multiprocessor_count;
    int max_threads_per_block;
    std::size_t shared_memory_per_block;
    std::string name;
};

}  // namespace runtime
}  // namespace tacker

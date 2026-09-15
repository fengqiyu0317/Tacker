#pragma once

#include "runtime/Kernel.h"

#include <functional>

namespace tacker {
namespace runtime {

// Scheduler-facing execution abstraction. Implementations must enqueue work and
// return immediately; eventComplete is the only operation used for progress.
// In particular, no implementation should call cudaDeviceSynchronize.
class ExecutionBackend {
public:
    virtual ~ExecutionBackend() {}

    virtual EventHandle launch(const KernelInvocation& invocation, StreamHandle stream) = 0;
    // Enqueues an opaque library operation (for example a CUB call) on the
    // supplied stream, then records a completion event behind it.
    virtual EventHandle enqueueOpaque(
        StreamHandle stream, const std::function<void(StreamHandle)>& operation) = 0;
    virtual EventHandle enqueueHostFence(StreamHandle stream,
                                         const std::function<void()>& callback) = 0;
    virtual bool eventComplete(EventHandle event) = 0;
    virtual void releaseEvent(EventHandle event) = 0;
    virtual DeviceProperties deviceProperties() const = 0;
};

}  // namespace runtime
}  // namespace tacker

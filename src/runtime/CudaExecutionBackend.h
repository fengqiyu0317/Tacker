#pragma once

#include "runtime/ExecutionBackend.h"

#include <memory>

namespace tacker {
namespace runtime {

// CUDA runtime API backend. StreamHandle and externally supplied EventHandle
// values are the uintptr_t representations of cudaStream_t/cudaEvent_t.
class CudaExecutionBackend : public ExecutionBackend {
public:
    explicit CudaExecutionBackend(int device_ordinal = 0);
    ~CudaExecutionBackend();

    EventHandle launch(const KernelInvocation& invocation, StreamHandle stream) override;
    EventHandle enqueueOpaque(StreamHandle stream,
                              const std::function<void(StreamHandle)>& operation) override;
    EventHandle enqueueHostFence(StreamHandle stream,
                                 const std::function<void()>& callback) override;
    bool eventComplete(EventHandle event) override;
    void releaseEvent(EventHandle event) override;
    DeviceProperties deviceProperties() const override;

    static StreamHandle wrapStream(void* cuda_stream);
    static EventHandle wrapExternalEvent(void* cuda_event);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace runtime
}  // namespace tacker

#include "runtime/CudaExecutionBackend.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace tacker {
namespace runtime {
namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

void checkDriver(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* name = NULL;
    const char* description = NULL;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &description);
    std::ostringstream message;
    message << operation << " failed: " << (name ? name : "unknown")
            << " (" << (description ? description : "no description") << ")";
    throw std::runtime_error(message.str());
}

cudaStream_t unwrapStream(StreamHandle stream) {
    return reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(stream));
}

cudaEvent_t unwrapEvent(EventHandle event) {
    return reinterpret_cast<cudaEvent_t>(static_cast<std::uintptr_t>(event.value));
}

EventHandle recordCompletion(cudaStream_t stream) {
    cudaEvent_t event = NULL;
    checkCuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags");
    try {
        checkCuda(cudaEventRecord(event, stream), "cudaEventRecord");
    } catch (...) {
        cudaEventDestroy(event);
        throw;
    }
    return EventHandle(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(event)));
}

#if CUDART_VERSION >= 10000
void CUDART_CB invokeHostCallback(void* raw_callback) {
    std::unique_ptr<std::function<void()> > callback(
        static_cast<std::function<void()>*>(raw_callback));
    try {
        if (*callback) (*callback)();
    } catch (...) {
        // Exceptions must never cross the CUDA callback ABI boundary.
    }
}
#else
void CUDART_CB invokeHostCallback(cudaStream_t, cudaError_t,
                                  void* raw_callback) {
    std::unique_ptr<std::function<void()> > callback(
        static_cast<std::function<void()>*>(raw_callback));
    try {
        if (*callback) (*callback)();
    } catch (...) {
    }
}
#endif

}  // namespace

struct CudaExecutionBackend::Impl {
    DeviceProperties properties;
};

CudaExecutionBackend::CudaExecutionBackend(int device_ordinal) : impl_(new Impl) {
    checkCuda(cudaSetDevice(device_ordinal), "cudaSetDevice");
    cudaDeviceProp property;
    checkCuda(cudaGetDeviceProperties(&property, device_ordinal), "cudaGetDeviceProperties");
    impl_->properties.device_ordinal = device_ordinal;
    impl_->properties.major = property.major;
    impl_->properties.minor = property.minor;
    impl_->properties.multiprocessor_count = property.multiProcessorCount;
    impl_->properties.max_threads_per_block = property.maxThreadsPerBlock;
    impl_->properties.shared_memory_per_block = property.sharedMemPerBlock;
    impl_->properties.name = property.name;
}

CudaExecutionBackend::~CudaExecutionBackend() {}

EventHandle CudaExecutionBackend::launch(const KernelInvocation& invocation,
                                         StreamHandle stream) {
    if (!invocation.spec || invocation.spec->native_handle == NULL) {
        throw std::invalid_argument("CUDA kernel invocation has no native function handle");
    }
    const Dim3 grid = invocation.launchGrid();
    const Dim3 block = invocation.launchBlock();
    std::vector<void*> arguments = invocation.arguments.argumentPointers();
    const dim3 cuda_grid(grid.x, grid.y, grid.z);
    const dim3 cuda_block(block.x, block.y, block.z);
    cudaStream_t cuda_stream = unwrapStream(stream);
    if (invocation.spec->launch_api == KernelLaunchApi::Driver) {
        checkDriver(cuLaunchKernel(reinterpret_cast<CUfunction>(invocation.spec->native_handle),
                                   grid.x, grid.y, grid.z,
                                   block.x, block.y, block.z,
                                   static_cast<unsigned int>(invocation.dynamicSharedMemory()),
                                   reinterpret_cast<CUstream>(cuda_stream),
                                   arguments.empty() ? NULL : arguments.data(), NULL),
                    "cuLaunchKernel");
    } else {
        checkCuda(cudaLaunchKernel(invocation.spec->native_handle, cuda_grid, cuda_block,
                                   arguments.empty() ? NULL : arguments.data(),
                                   invocation.dynamicSharedMemory(), cuda_stream),
                  "cudaLaunchKernel");
    }
    return recordCompletion(cuda_stream);
}

EventHandle CudaExecutionBackend::enqueueOpaque(
    StreamHandle stream, const std::function<void(StreamHandle)>& operation) {
    if (!operation) throw std::invalid_argument("opaque CUDA operation callback is required");
    operation(stream);
    return recordCompletion(unwrapStream(stream));
}

EventHandle CudaExecutionBackend::enqueueHostFence(
    StreamHandle stream, const std::function<void()>& callback) {
    cudaStream_t cuda_stream = unwrapStream(stream);
    std::unique_ptr<std::function<void()> > owned(new std::function<void()>(callback));
#if CUDART_VERSION >= 10000
    checkCuda(cudaLaunchHostFunc(cuda_stream, invokeHostCallback, owned.get()),
              "cudaLaunchHostFunc");
#else
    checkCuda(cudaStreamAddCallback(cuda_stream, invokeHostCallback, owned.get(), 0),
              "cudaStreamAddCallback");
#endif
    owned.release();
    return recordCompletion(cuda_stream);
}

bool CudaExecutionBackend::eventComplete(EventHandle event) {
    if (!event.valid()) throw std::invalid_argument("cannot query an invalid CUDA event");
    const cudaError_t status = cudaEventQuery(unwrapEvent(event));
    if (status == cudaSuccess) return true;
    if (status == cudaErrorNotReady) return false;
    checkCuda(status, "cudaEventQuery");
    return false;
}

void CudaExecutionBackend::releaseEvent(EventHandle event) {
    if (!event.valid()) return;
    checkCuda(cudaEventDestroy(unwrapEvent(event)), "cudaEventDestroy");
}

DeviceProperties CudaExecutionBackend::deviceProperties() const {
    return impl_->properties;
}

StreamHandle CudaExecutionBackend::wrapStream(void* cuda_stream) {
    return static_cast<StreamHandle>(reinterpret_cast<std::uintptr_t>(cuda_stream));
}

EventHandle CudaExecutionBackend::wrapExternalEvent(void* cuda_event) {
    return EventHandle(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(cuda_event)));
}

}  // namespace runtime
}  // namespace tacker

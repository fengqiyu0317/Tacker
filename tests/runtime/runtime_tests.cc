#include "runtime/ExecutionBackend.h"
#include "runtime/Registry.h"
#include "runtime/Scheduler.h"
#include "runtime/TaskGraph.h"

#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tacker::runtime;

namespace {

int failures = 0;

#define EXPECT_TRUE(expression) do {                                                \
    if (!(expression)) {                                                           \
        std::cerr << __FILE__ << ":" << __LINE__ << ": expected " #expression     \
                  << std::endl;                                                    \
        ++failures;                                                               \
    }                                                                             \
} while (0)

#define EXPECT_EQ(left, right) do {                                                \
    if (!((left) == (right))) {                                                    \
        std::cerr << __FILE__ << ":" << __LINE__ << ": expected " #left           \
                  << " == " #right << std::endl;                                  \
        ++failures;                                                               \
    }                                                                             \
} while (0)

template <typename Exception, typename Callback>
void expectThrows(const Callback& callback) {
    try {
        callback();
        ++failures;
        std::cerr << "expected exception was not thrown" << std::endl;
    } catch (const Exception&) {
    }
}

class FakeBackend : public ExecutionBackend {
public:
    FakeBackend() : next_event_(1), throw_mixed_once_(false) {
        properties_.major = 8;
        properties_.minor = 6;
        properties_.multiprocessor_count = 84;
        properties_.max_threads_per_block = 1024;
        properties_.shared_memory_per_block = 48 * 1024;
    }

    EventHandle launch(const KernelInvocation& invocation, StreamHandle stream) override {
        if (!invocation.spec) throw std::invalid_argument("missing spec");
        launched_keys.push_back(invocation.spec->key);
        launched_streams.push_back(stream);
        if (throw_mixed_once_ && invocation.spec->key == "mixed") {
            throw_mixed_once_ = false;
            throw std::runtime_error("synthetic mixed failure");
        }
        return makeEvent(0);
    }

    EventHandle enqueueHostFence(StreamHandle stream,
                                 const std::function<void()>& callback) override {
        launched_keys.push_back("host-fence");
        launched_streams.push_back(stream);
        if (callback) callback();
        return makeEvent(0);
    }

    EventHandle enqueueOpaque(
        StreamHandle stream, const std::function<void(StreamHandle)>& operation) override {
        launched_keys.push_back("opaque-callback");
        launched_streams.push_back(stream);
        if (operation) operation(stream);
        return makeEvent(0);
    }

    bool eventComplete(EventHandle event) override {
        std::map<std::uint64_t, int>::iterator it = delays_.find(event.value);
        if (it == delays_.end()) throw std::invalid_argument("unknown fake event");
        if (it->second > 0) {
            --it->second;
            return false;
        }
        return true;
    }

    void releaseEvent(EventHandle event) override { released.push_back(event.value); }
    DeviceProperties deviceProperties() const override { return properties_; }

    EventHandle makeEvent(int incomplete_queries) {
        const EventHandle event(next_event_++);
        delays_[event.value] = incomplete_queries;
        return event;
    }

    void throwMixedOnce() { throw_mixed_once_ = true; }

    std::vector<std::string> launched_keys;
    std::vector<StreamHandle> launched_streams;
    std::vector<std::uint64_t> released;

private:
    std::uint64_t next_event_;
    bool throw_mixed_once_;
    std::map<std::uint64_t, int> delays_;
    DeviceProperties properties_;
};

std::shared_ptr<KernelSpec> spec(const std::string& key, bool mixable = true) {
    std::shared_ptr<KernelSpec> result(new KernelSpec);
    result->key = key;
    result->symbol = key;
    result->architecture = "sm_86";
    result->block = Dim3(128);
    result->grid = Dim3(84);
    result->mixable = mixable;
    return result;
}

KernelInvocation invocation(const std::shared_ptr<const KernelSpec>& kernel) {
    KernelInvocation result(kernel);
    int scalar = 17;
    result.arguments.addScalar(scalar);
    return result;
}

void testArgumentPack() {
    OwnedArgumentPack first;
    int scalar = 42;
    first.addScalar(scalar);
    int pointee = 7;
    first.addPointer(&pointee);
    OwnedArgumentPack second = first;
    std::vector<void*> pointers = second.argumentPointers();
    int scalar_copy = 0;
    void* pointer_copy = NULL;
    std::memcpy(&scalar_copy, pointers[0], sizeof(scalar_copy));
    std::memcpy(&pointer_copy, pointers[1], sizeof(pointer_copy));
    EXPECT_EQ(scalar_copy, 42);
    EXPECT_TRUE(pointer_copy == &pointee);
}

void testTaskGraphAndCycleDetection() {
    TaskGraph graph("dag");
    const NodeId first = graph.addNode(NodeType::Solo, "first");
    const NodeId second = graph.addNode(NodeType::Opaque, "second");
    graph.addDependency(first, second);
    graph.finalize();
    EXPECT_EQ(graph.readyNodes().size(), static_cast<std::size_t>(1));
    graph.markSubmitted(first);
    graph.markCompleted(first);
    EXPECT_EQ(graph.readyNodes()[0], second);
    graph.markSubmitted(second);
    graph.markCompleted(second);
    EXPECT_TRUE(graph.complete());

    TaskGraph cyclic("cycle");
    const NodeId a = cyclic.addNode(NodeType::Solo, "a");
    const NodeId b = cyclic.addNode(NodeType::Solo, "b");
    cyclic.addDependency(a, b);
    cyclic.addDependency(b, a);
    expectThrows<std::invalid_argument>([&cyclic]() { cyclic.finalize(); });
}

void populateRegistries(KernelRegistry* kernels, ProfileRegistry* profiles,
                        double primary_slowdown_percent = 2.0,
                        bool register_profile = true,
                        const Dim3& mixed_block = Dim3(128),
                        const std::string& mixed_architecture = "sm_86") {
    kernels->registerKernel(spec("left"));
    kernels->registerKernel(spec("right"));
    kernels->registerKernel(spec("tail", false));
    MixedKernelRegistration registration;
    registration.left_key = "left";
    registration.right_key = "right";
    std::shared_ptr<KernelSpec> mixed = spec("mixed", false);
    mixed->block = mixed_block;
    mixed->architecture = mixed_architecture;
    registration.mixed_spec = mixed;
    kernels->registerMixed(registration);

    if (!register_profile) return;
    ProfileRecord profile;
    profile.enabled = true;
    profile.valid = true;
    profile.solo_sum_us = 10.0;
    profile.mixed_p50_us = 7.0;
    profile.mixed_p95_us = 8.0;
    profile.primary_slowdown_percent = primary_slowdown_percent;
    profile.median_throughput_fps = 100.0;
    profile.selected_for_deployment = true;
    profile.manifest_hash = "synthetic-sealed-manifest";
    profiles->set("left", "right", "mixed", profile);
}

void driveToIdle(Scheduler* scheduler) {
    for (int iteration = 0; iteration < 20 && !scheduler->idle(); ++iteration) {
        scheduler->progress();
    }
    EXPECT_TRUE(scheduler->idle());
}

void testMixedCompletionAdvancesBothGraphs() {
    KernelRegistry kernels;
    ProfileRegistry profiles;
    populateRegistries(&kernels, &profiles);
    FakeBackend backend;
    SchedulerOptions scheduler_options;
    scheduler_options.fusion_stream = 99;
    Scheduler scheduler(backend, kernels, profiles, scheduler_options);

    std::shared_ptr<TaskGraph> left_graph(new TaskGraph("raster"));
    const NodeId left = left_graph->addNode(NodeType::GPTB, "left",
                                            invocation(kernels.findKernel("left")));
    const NodeId left_tail = left_graph->addNode(NodeType::Solo, "tail",
                                                 invocation(kernels.findKernel("tail")));
    left_graph->addDependency(left, left_tail);
    left_graph->finalize();

    std::shared_ptr<TaskGraph> right_graph(new TaskGraph("deformation"));
    right_graph->addNode(NodeType::GPTB, "right",
                         invocation(kernels.findKernel("right")));
    right_graph->finalize();

    SubmissionOptions lc;
    lc.stream = 10;
    lc.role = SubmissionRole::LatencyCritical;
    SubmissionOptions be;
    be.stream = 20;
    be.role = SubmissionRole::BestEffort;
    const SubmissionId left_submission = scheduler.submit(left_graph, lc);
    const SubmissionId right_submission = scheduler.submit(right_graph, be);

    scheduler.progress();
    EXPECT_EQ(backend.launched_keys.size(), static_cast<std::size_t>(1));
    EXPECT_EQ(backend.launched_keys[0], std::string("mixed"));
    EXPECT_EQ(backend.launched_streams[0], static_cast<StreamHandle>(99));
    EXPECT_EQ(scheduler.inFlightCount(), static_cast<std::size_t>(1));

    driveToIdle(&scheduler);
    EXPECT_TRUE(scheduler.submissionComplete(left_submission));
    EXPECT_TRUE(scheduler.submissionComplete(right_submission));
    EXPECT_EQ(backend.launched_keys[1], std::string("tail"));
    EXPECT_EQ(backend.launched_streams[1], static_cast<StreamHandle>(10));
}

void testSoloFallback() {
    KernelRegistry kernels;
    ProfileRegistry profiles;
    populateRegistries(&kernels, &profiles);
    FakeBackend backend;
    backend.throwMixedOnce();
    Scheduler scheduler(backend, kernels, profiles);

    std::shared_ptr<TaskGraph> left_graph(new TaskGraph("left"));
    left_graph->addNode(NodeType::GPTB, "left", invocation(kernels.findKernel("left")));
    left_graph->finalize();
    std::shared_ptr<TaskGraph> right_graph(new TaskGraph("right"));
    right_graph->addNode(NodeType::GPTB, "right", invocation(kernels.findKernel("right")));
    right_graph->finalize();
    scheduler.submit(left_graph);
    scheduler.submit(right_graph);
    driveToIdle(&scheduler);
    EXPECT_EQ(backend.launched_keys.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(backend.launched_keys[0], std::string("mixed"));
    EXPECT_EQ(backend.launched_keys[1], std::string("left"));
    EXPECT_EQ(backend.launched_keys[2], std::string("right"));
    EXPECT_EQ(scheduler.diagnostics().size(), static_cast<std::size_t>(1));
}

void testExternalEventHostFenceAndOpaqueNode() {
    KernelRegistry kernels;
    ProfileRegistry profiles;
    FakeBackend backend;
    Scheduler scheduler(backend, kernels, profiles);
    bool callback_ran = false;
    bool opaque_ran = false;

    std::shared_ptr<TaskGraph> graph(new TaskGraph("special"));
    const NodeId external = graph->addExternalEventNode("external", backend.makeEvent(1));
    const NodeId fence = graph->addHostFenceNode("fence", [&callback_ran]() {
        callback_ran = true;
    });
    const NodeId opaque_node = graph->addOpaqueNode(
        "opaque", [&opaque_ran](StreamHandle) { opaque_ran = true; });
    graph->addDependency(external, fence);
    graph->addDependency(fence, opaque_node);
    graph->finalize();
    scheduler.submit(graph);
    driveToIdle(&scheduler);
    EXPECT_TRUE(callback_ran);
    EXPECT_TRUE(opaque_ran);
    EXPECT_EQ(backend.launched_keys[0], std::string("host-fence"));
    EXPECT_EQ(backend.launched_keys[1], std::string("opaque-callback"));
}

void testProfileGateUsesSolo() {
    KernelRegistry kernels;
    ProfileRegistry profiles;
    populateRegistries(&kernels, &profiles);
    profiles.disable("left", "right", "mixed");
    FakeBackend backend;
    Scheduler scheduler(backend, kernels, profiles);
    std::shared_ptr<TaskGraph> left_graph(new TaskGraph("left"));
    left_graph->addNode(NodeType::GPTB, "left", invocation(kernels.findKernel("left")));
    left_graph->finalize();
    std::shared_ptr<TaskGraph> right_graph(new TaskGraph("right"));
    right_graph->addNode(NodeType::GPTB, "right", invocation(kernels.findKernel("right")));
    right_graph->finalize();
    scheduler.submit(left_graph);
    scheduler.submit(right_graph);
    driveToIdle(&scheduler);
    EXPECT_EQ(backend.launched_keys.size(), static_cast<std::size_t>(2));
    EXPECT_EQ(backend.launched_keys[0], std::string("left"));
    EXPECT_EQ(backend.launched_keys[1], std::string("right"));
}

void expectAdmissionGateUsesSolo(double slowdown, bool register_profile,
                                 const Dim3& mixed_block,
                                 const std::string& mixed_architecture) {
    KernelRegistry kernels;
    ProfileRegistry profiles;
    populateRegistries(&kernels, &profiles, slowdown, register_profile,
                       mixed_block, mixed_architecture);
    FakeBackend backend;
    Scheduler scheduler(backend, kernels, profiles);

    std::shared_ptr<TaskGraph> left_graph(new TaskGraph("left"));
    left_graph->addNode(NodeType::GPTB, "left", invocation(kernels.findKernel("left")));
    left_graph->finalize();
    std::shared_ptr<TaskGraph> right_graph(new TaskGraph("right"));
    right_graph->addNode(NodeType::GPTB, "right", invocation(kernels.findKernel("right")));
    right_graph->finalize();
    scheduler.submit(left_graph);
    scheduler.submit(right_graph);
    driveToIdle(&scheduler);

    EXPECT_EQ(backend.launched_keys.size(), static_cast<std::size_t>(2));
    EXPECT_EQ(backend.launched_keys[0], std::string("left"));
    EXPECT_EQ(backend.launched_keys[1], std::string("right"));
}

void testAdmissionGatesUseSolo() {
    // Profiles are mandatory by default, and stale architecture/resource
    // manifests must not be launched on the current device.
    expectAdmissionGateUsesSolo(2.0, false, Dim3(384), "sm_86");
    expectAdmissionGateUsesSolo(2.0, true, Dim3(384), "sm_90");
    expectAdmissionGateUsesSolo(2.0, true, Dim3(2048), "sm_86");
}

void testRasterSlowdownAndSubmissionRolesAreDiagnosticOnly() {
    KernelRegistry kernels;
    ProfileRegistry profiles;
    populateRegistries(&kernels, &profiles, 250.0, true, Dim3(384), "sm_86");
    FakeBackend backend;
    Scheduler scheduler(backend, kernels, profiles);

    std::shared_ptr<TaskGraph> left_graph(new TaskGraph("left"));
    left_graph->addNode(NodeType::GPTB, "left", invocation(kernels.findKernel("left")));
    left_graph->finalize();
    std::shared_ptr<TaskGraph> right_graph(new TaskGraph("right"));
    right_graph->addNode(NodeType::GPTB, "right", invocation(kernels.findKernel("right")));
    right_graph->finalize();
    SubmissionOptions left_options;
    left_options.role = SubmissionRole::BestEffort;
    SubmissionOptions right_options;
    right_options.role = SubmissionRole::LatencyCritical;
    scheduler.submit(left_graph, left_options);
    scheduler.submit(right_graph, right_options);
    driveToIdle(&scheduler);
    EXPECT_EQ(backend.launched_keys[0], std::string("mixed"));
}

void testMultipleMixedRegistrationsUseSealedDeploymentWinner() {
    KernelRegistry kernels;
    ProfileRegistry profiles;
    populateRegistries(&kernels, &profiles);

    MixedKernelRegistration registration;
    registration.left_key = "left";
    registration.right_key = "right";
    std::shared_ptr<KernelSpec> mixed_fast = spec("mixed_fast", false);
    mixed_fast->block = Dim3(384);
    mixed_fast->architecture = "sm_86";
    registration.mixed_spec = mixed_fast;
    kernels.registerMixed(registration);
    EXPECT_EQ(kernels.findMixedCandidates("left", "right").size(),
              static_cast<std::size_t>(2));

    ProfileRecord old_profile;
    old_profile.enabled = true;
    old_profile.valid = true;
    old_profile.median_throughput_fps = 100.0;
    old_profile.selected_for_deployment = false;
    profiles.set("left", "right", "mixed", old_profile);
    ProfileRecord winner = old_profile;
    winner.median_throughput_fps = 120.0;
    winner.selected_for_deployment = true;
    winner.manifest_hash = "synthetic-fast-manifest";
    profiles.set("left", "right", "mixed_fast", winner);

    FakeBackend backend;
    Scheduler scheduler(backend, kernels, profiles);
    std::shared_ptr<TaskGraph> left_graph(new TaskGraph("left"));
    left_graph->addNode(NodeType::GPTB, "left", invocation(kernels.findKernel("left")));
    left_graph->finalize();
    std::shared_ptr<TaskGraph> right_graph(new TaskGraph("right"));
    right_graph->addNode(NodeType::GPTB, "right", invocation(kernels.findKernel("right")));
    right_graph->finalize();
    scheduler.submit(left_graph);
    scheduler.submit(right_graph);
    driveToIdle(&scheduler);
    EXPECT_EQ(backend.launched_keys[0], std::string("mixed_fast"));
}

}  // namespace

int main() {
    testArgumentPack();
    testTaskGraphAndCycleDetection();
    testMixedCompletionAdvancesBothGraphs();
    testSoloFallback();
    testExternalEventHostFenceAndOpaqueNode();
    testProfileGateUsesSolo();
    testAdmissionGatesUseSolo();
    testRasterSlowdownAndSubmissionRolesAreDiagnosticOnly();
    testMultipleMixedRegistrationsUseSealedDeploymentWinner();
    if (failures != 0) {
        std::cerr << failures << " runtime test(s) failed" << std::endl;
        return 1;
    }
    std::cout << "all tacker runtime tests passed" << std::endl;
    return 0;
}

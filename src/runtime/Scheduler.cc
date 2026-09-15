#include "runtime/Scheduler.h"

#include <algorithm>
#include <exception>
#include <sstream>
#include <stdexcept>

namespace tacker {
namespace runtime {
namespace {

std::string mixedPairKey(const std::string& left, const std::string& right,
                         const std::string& mixed) {
    return left + "\n" + right + "\n" + mixed;
}

}  // namespace

Scheduler::Scheduler(ExecutionBackend& backend, const KernelRegistry& kernels,
                     const ProfileRegistry& profiles, const SchedulerOptions& options)
    : backend_(backend), kernels_(kernels), profiles_(profiles), options_(options),
      next_submission_id_(1) {}

Scheduler::~Scheduler() {
    for (std::vector<InFlight>::const_iterator it = in_flight_.begin();
         it != in_flight_.end(); ++it) {
        if (!it->owns_event) continue;
        try {
            backend_.releaseEvent(it->event);
        } catch (...) {
            // Destructors must not throw. Backends may release pending events
            // asynchronously, so this still does not introduce a sync point.
        }
    }
}

Submission Scheduler::submit(const std::shared_ptr<TaskGraph>& graph,
                             const SubmissionOptions& options) {
    if (!graph) throw std::invalid_argument("cannot submit a null task graph");
    if (!graph->finalized()) throw std::invalid_argument("task graph must be finalized");
    for (std::map<SubmissionId, SubmissionState>::const_iterator it = submissions_.begin();
         it != submissions_.end(); ++it) {
        if (it->second.graph.get() == graph.get()) {
            throw std::invalid_argument("a TaskGraph instance can only be submitted once");
        }
    }
    graph->reset();
    SubmissionState state;
    state.id = next_submission_id_++;
    state.graph = graph;
    state.options = options;
    submissions_[state.id] = state;
    Submission result;
    result.id = state.id;
    result.graph = graph;
    result.options = options;
    return result;
}

Scheduler::SubmissionState& Scheduler::checkedSubmission(SubmissionId id) {
    std::map<SubmissionId, SubmissionState>::iterator it = submissions_.find(id);
    if (it == submissions_.end()) throw std::out_of_range("unknown submission id");
    return it->second;
}

const Scheduler::SubmissionState& Scheduler::checkedSubmission(SubmissionId id) const {
    std::map<SubmissionId, SubmissionState>::const_iterator it = submissions_.find(id);
    if (it == submissions_.end()) throw std::out_of_range("unknown submission id");
    return it->second;
}

void Scheduler::track(EventHandle event, bool owns_event, const std::vector<NodeRef>& nodes,
                      const KernelInvocation& invocation_lifetime) {
    if (!event.valid()) throw std::runtime_error("execution backend returned an invalid event");
    InFlight flight;
    flight.event = event;
    flight.owns_event = owns_event;
    flight.nodes = nodes;
    flight.invocation_lifetime = invocation_lifetime;
    in_flight_.push_back(flight);
}

bool Scheduler::pollCompletions() {
    bool changed = false;
    std::vector<InFlight> remaining;
    remaining.reserve(in_flight_.size());
    for (std::vector<InFlight>::iterator it = in_flight_.begin(); it != in_flight_.end(); ++it) {
        if (!backend_.eventComplete(it->event)) {
            remaining.push_back(*it);
            continue;
        }
        for (std::vector<NodeRef>::const_iterator node = it->nodes.begin();
             node != it->nodes.end(); ++node) {
            checkedSubmission(node->submission).graph->markCompleted(node->node);
        }
        if (it->owns_event) backend_.releaseEvent(it->event);
        changed = true;
    }
    in_flight_.swap(remaining);
    return changed;
}

bool Scheduler::architectureMatches(const KernelSpec& spec) const {
    if (spec.architecture.empty()) return true;
    const DeviceProperties device = backend_.deviceProperties();
    std::ostringstream actual;
    actual << "sm_" << device.major << device.minor;
    return spec.architecture == actual.str();
}

bool Scheduler::resourcesFit(const KernelSpec& spec) const {
    const DeviceProperties device = backend_.deviceProperties();
    const std::uint64_t threads = static_cast<std::uint64_t>(spec.block.x) *
                                  static_cast<std::uint64_t>(spec.block.y) *
                                  static_cast<std::uint64_t>(spec.block.z);
    if (device.max_threads_per_block > 0 &&
        threads > static_cast<std::uint64_t>(device.max_threads_per_block)) return false;
    if (device.shared_memory_per_block > 0) {
        if (spec.static_shared_memory > device.shared_memory_per_block) return false;
        if (spec.dynamic_shared_memory >
            device.shared_memory_per_block - spec.static_shared_memory) return false;
    }
    return true;
}

Scheduler::Candidate Scheduler::bestCandidate() const {
    Candidate best;
    for (std::map<SubmissionId, SubmissionState>::const_iterator left_submission =
             submissions_.begin(); left_submission != submissions_.end(); ++left_submission) {
        const std::vector<NodeId> left_ready = left_submission->second.graph->readyNodes();
        for (std::vector<NodeId>::const_iterator left_id = left_ready.begin();
             left_id != left_ready.end(); ++left_id) {
            const TaskNode& left_node = left_submission->second.graph->node(*left_id);
            if (left_node.type != NodeType::GPTB || !left_node.invocation.spec) continue;
            for (std::map<SubmissionId, SubmissionState>::const_iterator right_submission =
                     submissions_.begin(); right_submission != submissions_.end(); ++right_submission) {
                if (left_submission->first == right_submission->first) continue;
                const std::vector<NodeId> right_ready = right_submission->second.graph->readyNodes();
                for (std::vector<NodeId>::const_iterator right_id = right_ready.begin();
                     right_id != right_ready.end(); ++right_id) {
                    const TaskNode& right_node = right_submission->second.graph->node(*right_id);
                    if (right_node.type != NodeType::GPTB || !right_node.invocation.spec) continue;

                    const std::vector<MixedKernelRegistration> registrations =
                        kernels_.findMixedCandidates(left_node.invocation.spec->key,
                                                     right_node.invocation.spec->key);
                    for (std::vector<MixedKernelRegistration>::const_iterator registration =
                             registrations.begin(); registration != registrations.end();
                         ++registration) {
                        if (!registration->mixed_spec ||
                            !architectureMatches(*registration->mixed_spec) ||
                            !resourcesFit(*registration->mixed_spec)) continue;
                        if (disabled_mixed_pairs_.count(mixedPairKey(
                                left_node.invocation.spec->key,
                                right_node.invocation.spec->key,
                                registration->mixed_spec->key)) != 0) continue;

                        ProfileRecord profile;
                        const bool has_profile = profiles_.find(
                            left_node.invocation.spec->key,
                            right_node.invocation.spec->key,
                            registration->mixed_spec->key, &profile);
                        if (options_.require_profile && !has_profile) continue;
                        // Offline whole-run selection is authoritative. Raster
                        // slowdown and leaf savings remain diagnostics only.
                        if (has_profile && (!profile.enabled || !profile.valid ||
                                            !profile.selected_for_deployment ||
                                            profile.median_throughput_fps <= 0.0)) continue;

                        const double score = has_profile ? profile.median_throughput_fps : 0.0;
                        const bool earlier_tie = best.valid && score == best.score &&
                            (left_submission->first < best.left.submission ||
                             (left_submission->first == best.left.submission && *left_id < best.left.node) ||
                             (left_submission->first == best.left.submission && *left_id == best.left.node &&
                              right_submission->first < best.right.submission));
                        if (!best.valid || score > best.score || earlier_tie) {
                            best.valid = true;
                            best.score = score;
                            best.left.submission = left_submission->first;
                            best.left.node = *left_id;
                            best.right.submission = right_submission->first;
                            best.right.node = *right_id;
                            best.registration = *registration;
                        }
                    }
                }
            }
        }
    }
    return best;
}

KernelInvocation Scheduler::makeMixedInvocation(const Candidate& candidate) const {
    const TaskNode& left = checkedSubmission(candidate.left.submission).graph->node(candidate.left.node);
    const TaskNode& right = checkedSubmission(candidate.right.submission).graph->node(candidate.right.node);
    KernelInvocation mixed(candidate.registration.mixed_spec);
    if (candidate.registration.compose_arguments) {
        mixed.arguments = candidate.registration.compose_arguments(left.invocation, right.invocation);
    } else {
        mixed.arguments = left.invocation.arguments;
        mixed.arguments.append(right.invocation.arguments);
    }
    mixed.stream = options_.fusion_stream;
    return mixed;
}

bool Scheduler::scheduleSpecialNodes() {
    bool changed = false;
    for (std::map<SubmissionId, SubmissionState>::iterator submission = submissions_.begin();
         submission != submissions_.end(); ++submission) {
        const std::vector<NodeId> ready = submission->second.graph->readyNodes();
        for (std::vector<NodeId>::const_iterator id = ready.begin(); id != ready.end(); ++id) {
            TaskNode& node = submission->second.graph->node(*id);
            if (node.type == NodeType::ExternalEvent) {
                submission->second.graph->markSubmitted(*id);
                NodeRef ref;
                ref.submission = submission->first;
                ref.node = *id;
                track(node.external_event, false, std::vector<NodeRef>(1, ref));
                changed = true;
            } else if (node.type == NodeType::HostFence) {
                submission->second.graph->markSubmitted(*id);
                NodeRef ref;
                ref.submission = submission->first;
                ref.node = *id;
                EventHandle event = backend_.enqueueHostFence(submission->second.options.stream,
                                                               node.host_callback);
                track(event, true, std::vector<NodeRef>(1, ref));
                changed = true;
            } else if (node.type == NodeType::Opaque && node.opaque_operation) {
                submission->second.graph->markSubmitted(*id);
                NodeRef ref;
                ref.submission = submission->first;
                ref.node = *id;
                EventHandle event = backend_.enqueueOpaque(submission->second.options.stream,
                                                            node.opaque_operation);
                track(event, true, std::vector<NodeRef>(1, ref));
                changed = true;
            }
        }
    }
    return changed;
}

bool Scheduler::scheduleMixedNodes() {
    bool changed = false;
    while (true) {
        const Candidate candidate = bestCandidate();
        if (!candidate.valid) break;
        SubmissionState& left = checkedSubmission(candidate.left.submission);
        SubmissionState& right = checkedSubmission(candidate.right.submission);
        const KernelInvocation mixed = makeMixedInvocation(candidate);
        try {
            const EventHandle event = backend_.launch(mixed, options_.fusion_stream);
            if (!event.valid()) throw std::runtime_error("backend returned an invalid event");
            left.graph->markSubmitted(candidate.left.node);
            right.graph->markSubmitted(candidate.right.node);
            std::vector<NodeRef> refs;
            refs.push_back(candidate.left);
            refs.push_back(candidate.right);
            track(event, true, refs, mixed);
            changed = true;
        } catch (const std::exception& error) {
            std::ostringstream message;
            message << "mixed launch " << candidate.registration.mixed_spec->key
                    << " failed; falling back to solo: " << error.what();
            diagnostics_.push_back(message.str());
            const TaskNode& left_node = left.graph->node(candidate.left.node);
            const TaskNode& right_node = right.graph->node(candidate.right.node);
            disabled_mixed_pairs_.insert(mixedPairKey(left_node.invocation.spec->key,
                                                       right_node.invocation.spec->key,
                                                       candidate.registration.mixed_spec->key));
            // Disable this pair for this Scheduler instance by scheduling both
            // ready nodes solo immediately. The registry remains immutable.
            const EventHandle left_event = backend_.launch(left_node.invocation, left.options.stream);
            if (!left_event.valid()) throw std::runtime_error("solo fallback returned an invalid event");
            left.graph->markSubmitted(candidate.left.node);
            track(left_event, true, std::vector<NodeRef>(1, candidate.left), left_node.invocation);
            const EventHandle right_event = backend_.launch(right_node.invocation, right.options.stream);
            if (!right_event.valid()) throw std::runtime_error("solo fallback returned an invalid event");
            right.graph->markSubmitted(candidate.right.node);
            track(right_event, true, std::vector<NodeRef>(1, candidate.right), right_node.invocation);
            changed = true;
        }
    }
    return changed;
}

bool Scheduler::scheduleSoloNodes() {
    bool changed = false;
    for (std::map<SubmissionId, SubmissionState>::iterator submission = submissions_.begin();
         submission != submissions_.end(); ++submission) {
        const std::vector<NodeId> ready = submission->second.graph->readyNodes();
        for (std::vector<NodeId>::const_iterator id = ready.begin(); id != ready.end(); ++id) {
            TaskNode& node = submission->second.graph->node(*id);
            if (node.type == NodeType::ExternalEvent || node.type == NodeType::HostFence ||
                (node.type == NodeType::Opaque && node.opaque_operation)) continue;
            if (!node.invocation.spec) {
                throw std::invalid_argument("kernel task node has no KernelSpec");
            }
            const StreamHandle stream = node.invocation.stream != 0
                ? node.invocation.stream : submission->second.options.stream;
            const EventHandle event = backend_.launch(node.invocation, stream);
            if (!event.valid()) throw std::runtime_error("solo launch returned an invalid event");
            submission->second.graph->markSubmitted(*id);
            NodeRef ref;
            ref.submission = submission->first;
            ref.node = *id;
            track(event, true, std::vector<NodeRef>(1, ref), node.invocation);
            changed = true;
        }
    }
    return changed;
}

bool Scheduler::progress() {
    bool changed = pollCompletions();
    changed = scheduleSpecialNodes() || changed;
    changed = scheduleMixedNodes() || changed;
    changed = scheduleSoloNodes() || changed;
    return changed;
}

bool Scheduler::submissionComplete(SubmissionId id) const {
    return checkedSubmission(id).graph->complete();
}

bool Scheduler::idle() const {
    if (!in_flight_.empty()) return false;
    for (std::map<SubmissionId, SubmissionState>::const_iterator it = submissions_.begin();
         it != submissions_.end(); ++it) {
        if (!it->second.graph->complete()) return false;
    }
    return true;
}

}  // namespace runtime
}  // namespace tacker

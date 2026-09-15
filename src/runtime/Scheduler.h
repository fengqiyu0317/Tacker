#pragma once

#include "runtime/ExecutionBackend.h"
#include "runtime/Registry.h"
#include "runtime/TaskGraph.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace tacker {
namespace runtime {

struct SchedulerOptions {
    SchedulerOptions()
        : fusion_stream(0), max_primary_slowdown_percent(5.0),
          minimum_savings_us(0.0), require_profile(true) {}

    StreamHandle fusion_stream;
    // Deprecated compatibility knobs. QoS-free scheduling records these
    // diagnostics but never uses them to reject an offline-selected profile.
    double max_primary_slowdown_percent;
    double minimum_savings_us;
    bool require_profile;
};

struct SubmissionOptions {
    SubmissionOptions() : stream(0), role(SubmissionRole::BestEffort) {}
    StreamHandle stream;
    SubmissionRole role;
};

struct Submission {
    Submission() : id(0) {}
    bool valid() const { return id != 0 && graph.get() != NULL; }
    operator SubmissionId() const { return id; }

    SubmissionId id;
    std::shared_ptr<TaskGraph> graph;
    SubmissionOptions options;
};

class Scheduler {
public:
    Scheduler(ExecutionBackend& backend, const KernelRegistry& kernels,
              const ProfileRegistry& profiles,
              const SchedulerOptions& options = SchedulerOptions());
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    Submission submit(const std::shared_ptr<TaskGraph>& graph,
                      const SubmissionOptions& options = SubmissionOptions());

    // Polls events and enqueues every currently ready operation. Returns true
    // if any node changed state or any work was enqueued.
    bool progress();
    bool submissionComplete(SubmissionId id) const;
    bool submissionComplete(const Submission& submission) const {
        return submissionComplete(submission.id);
    }
    bool idle() const;
    std::size_t inFlightCount() const { return in_flight_.size(); }
    const std::vector<std::string>& diagnostics() const { return diagnostics_; }

private:
    struct SubmissionState {
        SubmissionId id;
        std::shared_ptr<TaskGraph> graph;
        SubmissionOptions options;
    };

    struct NodeRef {
        SubmissionId submission;
        NodeId node;
    };

    struct InFlight {
        EventHandle event;
        bool owns_event;
        std::vector<NodeRef> nodes;
        KernelInvocation invocation_lifetime;
    };

    struct Candidate {
        Candidate() : valid(false), score(0.0) {}
        bool valid;
        double score;
        NodeRef left;
        NodeRef right;
        MixedKernelRegistration registration;
    };

    SubmissionState& checkedSubmission(SubmissionId id);
    const SubmissionState& checkedSubmission(SubmissionId id) const;
    bool pollCompletions();
    bool scheduleSpecialNodes();
    bool scheduleMixedNodes();
    bool scheduleSoloNodes();
    Candidate bestCandidate() const;
    bool architectureMatches(const KernelSpec& spec) const;
    bool resourcesFit(const KernelSpec& spec) const;
    KernelInvocation makeMixedInvocation(const Candidate& candidate) const;
    void track(EventHandle event, bool owns_event, const std::vector<NodeRef>& nodes,
               const KernelInvocation& invocation_lifetime = KernelInvocation());

    ExecutionBackend& backend_;
    const KernelRegistry& kernels_;
    const ProfileRegistry& profiles_;
    SchedulerOptions options_;
    std::map<SubmissionId, SubmissionState> submissions_;
    std::vector<InFlight> in_flight_;
    SubmissionId next_submission_id_;
    std::vector<std::string> diagnostics_;
    std::set<std::string> disabled_mixed_pairs_;
};

}  // namespace runtime
}  // namespace tacker

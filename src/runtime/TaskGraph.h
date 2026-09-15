#pragma once

#include "runtime/Kernel.h"

#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace tacker {
namespace runtime {

struct TaskNode {
    TaskNode()
        : id(0), type(NodeType::Solo), state(NodeState::Pending),
          remaining_predecessors(0) {}

    NodeId id;
    std::string name;
    NodeType type;
    NodeState state;
    KernelInvocation invocation;
    EventHandle external_event;
    std::function<void(StreamHandle)> opaque_operation;
    std::function<void()> host_callback;
    std::vector<NodeId> predecessors;
    std::vector<NodeId> successors;
    std::size_t remaining_predecessors;
};

class TaskGraph {
public:
    explicit TaskGraph(const std::string& name = std::string());

    NodeId addNode(NodeType type, const std::string& name,
                   const KernelInvocation& invocation = KernelInvocation());
    NodeId addExternalEventNode(const std::string& name, EventHandle event);
    NodeId addOpaqueNode(const std::string& name,
                         const std::function<void(StreamHandle)>& operation);
    NodeId addHostFenceNode(const std::string& name, const std::function<void()>& callback);
    void addDependency(NodeId predecessor, NodeId successor);

    // Validates that the graph is acyclic and initializes dependency counts.
    // A graph must be finalized before it can be submitted to a Scheduler.
    void finalize();
    void reset();

    bool finalized() const { return finalized_; }
    bool complete() const;
    std::size_t size() const { return nodes_.size(); }
    const std::string& name() const { return name_; }

    std::vector<NodeId> readyNodes() const;
    const TaskNode& node(NodeId id) const;
    TaskNode& node(NodeId id);
    void markSubmitted(NodeId id);
    void markCompleted(NodeId id);

private:
    TaskNode& checkedNode(NodeId id);
    const TaskNode& checkedNode(NodeId id) const;
    void requireMutable() const;

    std::string name_;
    std::map<NodeId, TaskNode> nodes_;
    NodeId next_id_;
    bool finalized_;
};

}  // namespace runtime
}  // namespace tacker

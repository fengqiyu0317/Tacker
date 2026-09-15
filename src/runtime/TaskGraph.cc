#include "runtime/TaskGraph.h"

#include <algorithm>
#include <deque>
#include <sstream>

namespace tacker {
namespace runtime {

TaskGraph::TaskGraph(const std::string& name)
    : name_(name), next_id_(1), finalized_(false) {}

void TaskGraph::requireMutable() const {
    if (finalized_) throw std::logic_error("cannot mutate a finalized task graph");
}

NodeId TaskGraph::addNode(NodeType type, const std::string& name,
                          const KernelInvocation& invocation) {
    requireMutable();
    TaskNode node;
    node.id = next_id_++;
    node.name = name;
    node.type = type;
    node.invocation = invocation;
    nodes_.insert(std::make_pair(node.id, node));
    return node.id;
}

NodeId TaskGraph::addExternalEventNode(const std::string& name, EventHandle event) {
    if (!event.valid()) throw std::invalid_argument("external event must be valid");
    const NodeId id = addNode(NodeType::ExternalEvent, name);
    nodes_[id].external_event = event;
    return id;
}

NodeId TaskGraph::addOpaqueNode(
    const std::string& name, const std::function<void(StreamHandle)>& operation) {
    if (!operation) throw std::invalid_argument("opaque operation callback is required");
    const NodeId id = addNode(NodeType::Opaque, name);
    nodes_[id].opaque_operation = operation;
    return id;
}

NodeId TaskGraph::addHostFenceNode(const std::string& name,
                                   const std::function<void()>& callback) {
    const NodeId id = addNode(NodeType::HostFence, name);
    nodes_[id].host_callback = callback;
    return id;
}

void TaskGraph::addDependency(NodeId predecessor, NodeId successor) {
    requireMutable();
    if (predecessor == successor) throw std::invalid_argument("a node cannot depend on itself");
    TaskNode& before = checkedNode(predecessor);
    TaskNode& after = checkedNode(successor);
    if (std::find(before.successors.begin(), before.successors.end(), successor) !=
        before.successors.end()) {
        return;
    }
    before.successors.push_back(successor);
    after.predecessors.push_back(predecessor);
}

void TaskGraph::finalize() {
    if (finalized_) return;

    std::map<NodeId, std::size_t> indegree;
    std::deque<NodeId> queue;
    for (std::map<NodeId, TaskNode>::const_iterator it = nodes_.begin(); it != nodes_.end(); ++it) {
        indegree[it->first] = it->second.predecessors.size();
        if (it->second.predecessors.empty()) queue.push_back(it->first);
    }

    std::size_t visited = 0;
    while (!queue.empty()) {
        const NodeId id = queue.front();
        queue.pop_front();
        ++visited;
        const TaskNode& current = checkedNode(id);
        for (std::vector<NodeId>::const_iterator it = current.successors.begin();
             it != current.successors.end(); ++it) {
            std::size_t& degree = indegree[*it];
            if (--degree == 0) queue.push_back(*it);
        }
    }

    if (visited != nodes_.size()) {
        throw std::invalid_argument("task graph contains a cycle");
    }

    finalized_ = true;
    reset();
}

void TaskGraph::reset() {
    if (!finalized_) throw std::logic_error("cannot reset an unfinalized task graph");
    for (std::map<NodeId, TaskNode>::iterator it = nodes_.begin(); it != nodes_.end(); ++it) {
        TaskNode& node = it->second;
        node.remaining_predecessors = node.predecessors.size();
        node.state = node.remaining_predecessors == 0 ? NodeState::Ready : NodeState::Pending;
    }
}

bool TaskGraph::complete() const {
    if (!finalized_) return false;
    for (std::map<NodeId, TaskNode>::const_iterator it = nodes_.begin(); it != nodes_.end(); ++it) {
        if (it->second.state != NodeState::Completed) return false;
    }
    return true;
}

std::vector<NodeId> TaskGraph::readyNodes() const {
    if (!finalized_) throw std::logic_error("task graph must be finalized first");
    std::vector<NodeId> result;
    for (std::map<NodeId, TaskNode>::const_iterator it = nodes_.begin(); it != nodes_.end(); ++it) {
        if (it->second.state == NodeState::Ready) result.push_back(it->first);
    }
    return result;
}

const TaskNode& TaskGraph::checkedNode(NodeId id) const {
    std::map<NodeId, TaskNode>::const_iterator it = nodes_.find(id);
    if (it == nodes_.end()) throw std::out_of_range("unknown task node id");
    return it->second;
}

TaskNode& TaskGraph::checkedNode(NodeId id) {
    std::map<NodeId, TaskNode>::iterator it = nodes_.find(id);
    if (it == nodes_.end()) throw std::out_of_range("unknown task node id");
    return it->second;
}

const TaskNode& TaskGraph::node(NodeId id) const { return checkedNode(id); }
TaskNode& TaskGraph::node(NodeId id) { return checkedNode(id); }

void TaskGraph::markSubmitted(NodeId id) {
    TaskNode& target = checkedNode(id);
    if (target.state != NodeState::Ready) {
        throw std::logic_error("only a ready task node can be submitted");
    }
    target.state = NodeState::Submitted;
}

void TaskGraph::markCompleted(NodeId id) {
    TaskNode& target = checkedNode(id);
    if (target.state != NodeState::Submitted && target.state != NodeState::Ready) {
        throw std::logic_error("only a ready or submitted task node can complete");
    }
    target.state = NodeState::Completed;
    for (std::vector<NodeId>::const_iterator it = target.successors.begin();
         it != target.successors.end(); ++it) {
        TaskNode& successor = checkedNode(*it);
        if (successor.remaining_predecessors == 0) {
            throw std::logic_error("task graph predecessor count underflow");
        }
        --successor.remaining_predecessors;
        if (successor.remaining_predecessors == 0) successor.state = NodeState::Ready;
    }
}

}  // namespace runtime
}  // namespace tacker

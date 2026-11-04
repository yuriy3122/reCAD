#include "GraphCut.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>

namespace recad {
namespace {

class ResidualNetwork {
    struct Arc { std::size_t to, reverse; double capacity; };
    std::vector<std::vector<Arc>> arcs_;
    static constexpr double epsilon = 1.0e-9;
public:
    explicit ResidualNetwork(std::size_t count) : arcs_(count) {}

    void add(std::size_t from, std::size_t to, double forward, double backward = 0.0) {
        const auto a = arcs_[from].size(), b = arcs_[to].size();
        arcs_[from].push_back({to, b, forward});
        arcs_[to].push_back({from, a, backward});
    }

    std::vector<bool> cut(std::size_t source, std::size_t sink) {
        const auto count = arcs_.size();
        std::vector<std::size_t> height(count, 0), current(count, 0);
        std::vector<double> excess(count, 0.0);
        std::vector<bool> queued(count, false);
        std::deque<std::size_t> active;
        const auto enqueue = [&](std::size_t vertex) {
            if (vertex != source && vertex != sink && !queued[vertex] && excess[vertex] > epsilon) {
                active.push_back(vertex); queued[vertex] = true;
            }
        };
        height[source] = count;
        for (auto& arc : arcs_[source]) {
            const double amount = arc.capacity;
            arc.capacity = 0.0;
            arcs_[arc.to][arc.reverse].capacity += amount;
            excess[arc.to] += amount;
            excess[source] -= amount;
            enqueue(arc.to);
        }
        const auto relabelAll = [&]() {
            std::fill(height.begin(), height.end(), count + 1);
            std::fill(current.begin(), current.end(), 0);
            std::vector<std::size_t> queue{sink};
            height[sink] = 0;
            for (std::size_t head = 0; head < queue.size(); ++head) {
                const auto vertex = queue[head];
                for (const auto& arc : arcs_[vertex]) {
                    if (arc.to != source && height[arc.to] == count + 1 &&
                        arcs_[arc.to][arc.reverse].capacity > epsilon) {
                        height[arc.to] = height[vertex] + 1;
                        queue.push_back(arc.to);
                    }
                }
            }
            height[source] = count;
        };
        relabelAll();
        std::size_t discharged = 0;
        while (!active.empty()) {
            const auto vertex = active.front(); active.pop_front(); queued[vertex] = false;
            while (excess[vertex] > epsilon) {
                if (current[vertex] == arcs_[vertex].size()) {
                    std::size_t minimum = std::numeric_limits<std::size_t>::max();
                    for (const auto& arc : arcs_[vertex]) {
                        if (arc.capacity > epsilon) minimum = std::min(minimum, height[arc.to]);
                    }
                    if (minimum == std::numeric_limits<std::size_t>::max()) break;
                    height[vertex] = minimum + 1;
                    current[vertex] = 0;
                    continue;
                }
                auto& arc = arcs_[vertex][current[vertex]];
                if (arc.capacity > epsilon && height[vertex] == height[arc.to] + 1) {
                    const double amount = std::min(excess[vertex], arc.capacity);
                    arc.capacity -= amount;
                    arcs_[arc.to][arc.reverse].capacity += amount;
                    excess[vertex] -= amount;
                    excess[arc.to] += amount;
                    enqueue(arc.to);
                } else ++current[vertex];
            }
            if (++discharged % count == 0) relabelAll();
        }
        std::vector<bool> reached(count, false);
        std::vector<std::size_t> queue{source}; reached[source] = true;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            for (const auto& arc : arcs_[queue[head]]) {
                if (arc.capacity > epsilon && !reached[arc.to]) {
                    reached[arc.to] = true; queue.push_back(arc.to);
                }
            }
        }
        return reached;
    }
};

} // namespace

std::vector<int> minimizePottsEnergy(
    std::span<const double> costs, std::size_t labelCount,
    std::span<const PottsEdge> edges, std::span<const int> initialLabels,
    std::size_t maximumSweeps) {
    const auto count = initialLabels.size();
    if (labelCount == 0 || costs.size() / labelCount != count || costs.size() % labelCount != 0)
        throw std::invalid_argument("Invalid Potts cost matrix");
    std::vector<int> labels(initialLabels.begin(), initialLabels.end());
    double hard = 1.0;
    for (const double value : costs) {
        if (std::isnan(value) || value < 0.0) throw std::invalid_argument("Invalid Potts unary cost");
        if (std::isfinite(value)) hard += value;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (labels[i] >= static_cast<int>(labelCount) ||
            (labels[i] >= 0 && !std::isfinite(costs[i * labelCount + labels[i]])))
            throw std::invalid_argument("Invalid initial Potts label");
    }
    for (const auto& edge : edges) {
        if (edge.first >= count || edge.second >= count || edge.first == edge.second ||
            !std::isfinite(edge.weight) || edge.weight < 0.0) throw std::invalid_argument("Invalid Potts edge");
        hard += edge.weight;
    }
    if (!std::isfinite(hard)) throw std::invalid_argument("Potts energy overflow");
    const auto energy = [&](const std::vector<int>& assignment) {
        double result = 0.0;
        for (std::size_t i = 0; i < count; ++i) if (assignment[i] >= 0) result += costs[i * labelCount + assignment[i]];
        for (const auto& edge : edges) {
            if (assignment[edge.first] >= 0 && assignment[edge.second] >= 0 &&
                assignment[edge.first] != assignment[edge.second]) result += edge.weight;
        }
        return result;
    };
    double previousEnergy = energy(labels);
    for (std::size_t sweep = 0; sweep < maximumSweeps; ++sweep) {
        bool improved = false;
        for (std::size_t alpha = 0; alpha < labelCount; ++alpha) {
            bool available = false;
            for (std::size_t i = 0; i < count && !available; ++i) {
                available = labels[i] >= 0 && labels[i] != static_cast<int>(alpha) &&
                    std::isfinite(costs[i * labelCount + alpha]);
            }
            if (!available) continue;
            std::vector<double> keep(count, 0.0), change(count, hard);
            for (std::size_t i = 0; i < count; ++i) {
                if (labels[i] < 0) continue;
                keep[i] = costs[i * labelCount + labels[i]];
                const double value = costs[i * labelCount + alpha];
                if (std::isfinite(value)) change[i] = value;
            }
            ResidualNetwork graph(count + 2);
            for (const auto& edge : edges) {
                const auto a = edge.first, b = edge.second;
                if (labels[a] < 0 || labels[b] < 0) continue;
                const double e00 = labels[a] != labels[b] ? edge.weight : 0.0;
                const double e01 = labels[a] != static_cast<int>(alpha) ? edge.weight : 0.0;
                const double e10 = labels[b] != static_cast<int>(alpha) ? edge.weight : 0.0;
                const double capacity = std::max(0.0, (e01 + e10 - e00) * 0.5);
                change[a] += e10 - e00 - capacity;
                change[b] += e01 - e00 - capacity;
                graph.add(a, b, capacity, capacity);
            }
            for (std::size_t i = 0; i < count; ++i) {
                const double offset = std::min(keep[i], change[i]);
                graph.add(count, i, std::max(0.0, keep[i] - offset));
                graph.add(i, count + 1, std::max(0.0, change[i] - offset));
            }
            const auto sourceSide = graph.cut(count, count + 1);
            auto candidate = labels;
            for (std::size_t i = 0; i < count; ++i) if (labels[i] >= 0 && sourceSide[i]) candidate[i] = static_cast<int>(alpha);
            const double candidateEnergy = energy(candidate);
            if (candidateEnergy < previousEnergy - 1.0e-10 * std::max(1.0, previousEnergy)) {
                labels = std::move(candidate); previousEnergy = candidateEnergy; improved = true;
            }
        }
        if (!improved) break;
    }
    return labels;
}

} // namespace recad

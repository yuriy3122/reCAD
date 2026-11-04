#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace recad {

struct PottsEdge {
    std::size_t first{};
    std::size_t second{};
    double weight{};
};

// Boykov/Veksler/Zabih alpha expansion for a metric Potts energy. Unary
// costs are node-major; infinity forbids an assignment. Negative initial
// labels exclude nodes. The residual-network solver is iterative push/relabel.
[[nodiscard]] std::vector<int> minimizePottsEnergy(
    std::span<const double> costs, std::size_t labelCount,
    std::span<const PottsEdge> edges, std::span<const int> initialLabels,
    std::size_t maximumSweeps = 16);

} // namespace recad

// Exact longest substring matches against streamed, separate training blocks.
// stdin: uint32 width, query_rows, query IDs, then [row_count, IDs]*, ending in 0.
// stdout: uint32 match length for every query token (same row-major order).
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

struct State {
    int length = 0, link = -1, best = 0;
    std::unordered_map<int, int> next;
};

uint32_t read_word() {
    uint32_t value;
    if (!std::cin.read(reinterpret_cast<char*>(&value), sizeof(value)))
        throw std::runtime_error("truncated input");
    return value;
}

int main() {
    try {
        std::ios::sync_with_stdio(false);
        const auto width = read_word(), rows = read_word();
        if (!width || !rows || uint64_t(width) * rows > 1000000)
            throw std::runtime_error("invalid query size");
        std::vector<State> states;
        states.reserve(2 * (uint64_t(width) + 1) * rows + 1);
        states.emplace_back();
        std::vector<int> endpoints;
        int last = 0;
        auto extend = [&](int token) {
            int current = int(states.size());
            states.emplace_back();
            states[current].length = states[last].length + 1;
            int p = last;
            while (p != -1 && !states[p].next.count(token)) {
                states[p].next[token] = current;
                p = states[p].link;
            }
            if (p == -1) states[current].link = 0;
            else {
                int q = states[p].next.at(token);
                if (states[p].length + 1 == states[q].length)
                    states[current].link = q;
                else {
                    int clone = int(states.size());
                    states.push_back(states[q]);
                    states[clone].length = states[p].length + 1;
                    states[clone].best = 0;
                    while (p != -1) {
                        auto it = states[p].next.find(token);
                        if (it == states[p].next.end() || it->second != q) break;
                        it->second = clone;
                        p = states[p].link;
                    }
                    states[q].link = states[current].link = clone;
                }
            }
            last = current;
        };
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < width; ++c) {
                const auto token = read_word();
                if (token > 2147483647) throw std::runtime_error("invalid token");
                extend(int(token));
                endpoints.push_back(last);
            }
            extend(-1 - int(r)); // unique separators, never present in training
        }
        while (const auto batch_rows = read_word()) {
            if (batch_rows > 4096) throw std::runtime_error("oversized batch");
            for (uint32_t r = 0; r < batch_rows; ++r) {
                int state = 0, length = 0;
                for (uint32_t c = 0; c < width; ++c) {
                    auto token = read_word();
                    if (token > 2147483647) throw std::runtime_error("invalid token");
                    while (state && !states[state].next.count(int(token))) {
                        state = states[state].link;
                        length = std::min(length, states[state].length);
                    }
                    auto it = states[state].next.find(int(token));
                    if (it != states[state].next.end()) {
                        state = it->second;
                        ++length;
                        states[state].best = std::max(states[state].best, length);
                    } else { state = 0; length = 0; }
                }
            }
        }
        std::vector<int> order(states.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return states[a].length < states[b].length;
        });
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const int v = *it, parent = states[v].link;
            if (parent >= 0)
                states[parent].best = std::max(states[parent].best,
                    std::min(states[v].best, states[parent].length));
        }
        for (int v : order) {
            const int parent = states[v].link;
            if (parent >= 0)
                states[v].best = std::max(states[v].best, states[parent].best);
        }
        for (int v : endpoints) {
            uint32_t length = uint32_t(states[v].best);
            std::cout.write(reinterpret_cast<const char*>(&length), sizeof(length));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "sol.h"

using namespace std;


float l2_distance(const float* a, const float* b, int d) {
    float dist = 0.0;
    for (int i = 0; i < d; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(dist);
}

void generate_data(vector<float>& base, vector<float>& queries, int num_vectors, int dimension) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    base.resize(num_vectors * dimension);
    queries.resize(10 * dimension); // Assuming you want to search with 10 query vectors

    for (auto& val : base) {
        val = dis(gen);
    }

    for (auto& val : queries) {
        val = dis(gen);
    }
}

float compute_recall(const vector<int>& search_result, const vector<int>& true_result) {
    int relevant = 0;
    for (int idx : search_result) {
        if (find(true_result.begin(), true_result.end(), idx) != true_result.end()) {
            relevant++;
        }
    }
    return static_cast<float>(relevant) / true_result.size();
}

int main() {
    int num_vectors = 100000;
    int dimension = 1024;

    vector<float> base;
    vector<float> queries;

    // Generate test data
    generate_data(base, queries, num_vectors, dimension);

    Solution solution;
    solution.build(dimension, base);

    float total_recall = 0.0;
    int k = 10;  // Assuming we are interested in the top 10 neighbors
    for (int i = 0; i < 10; ++i) { // Assuming 10 queries
        const float* query = &queries[i * dimension];
        vector<int> search_result(k);
        solution.search(vector<float>(query, query + dimension), search_result.data());

        // Compute the true nearest neighbors (brute-force)
        vector<pair<float, int>> distances;
        for (int j = 0; j < num_vectors; ++j) {
            const float* base_vec = &base[j * dimension];
            float dist = l2_distance(query, base_vec, dimension);
            distances.push_back({ dist, j });
        }
        sort(distances.begin(), distances.end());
        vector<int> true_result;
        for (int j = 0; j < k; ++j) {
            true_result.push_back(distances[j].second);
        }

        // Calculate recall
        float recall = compute_recall(search_result, true_result);
        total_recall += recall;
    }

    cout << "Average Recall: " << total_recall / 10 << endl;

    return 0;
}

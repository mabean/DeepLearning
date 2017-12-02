#include "Layer.h"

#include <cstdlib>
#include <ctime>
#include <omp.h>

Layer::Layer(const int size, const int previousLayerSize,
             const Activation &function,
             const Derivative &derivativeFunc)
    : m_values(size + 1)
    , m_weights(size)                   // weights from previous layer to this layer
    , m_errors(size)                    // errors in this layer nodes
    , m_function(function)
    , m_derivative(derivativeFunc)
{
    m_values[0] = 1.;
    std::srand(unsigned(std::time(0)));
    for (auto &vector : m_weights)
    {
        vector.resize(previousLayerSize + 1);
        vector[0] = 0;
        for (int i = 1; i < vector.size(); i++)
        {
            vector[i] = std::rand() / static_cast<double>(RAND_MAX) - 0.5;
        }
    }
}

int Layer::size() const
{
    return static_cast<int>(m_values.size());
}

std::vector<double> Layer::weights(const int nodeIndex) const
{
    return m_weights.at(nodeIndex);
}

std::vector<double> Layer::errors() const
{
    return m_errors;
}

std::vector<double> Layer::values() const
{
    return m_values;
}

void Layer::activateValues(const std::vector<double> &sums)
{
    for (int i = 0; i < sums.size(); i++)
    {
        m_values[i + 1] = m_function(sums[i]);
    }
}

void Layer::recomputeWeights(const std::shared_ptr<Layer> &nextLayer,
                             const std::vector<double> &previousValues,
                             const double learnRate)
{
    auto nextLayerErrors = nextLayer->errors();
#pragma omp parallel for
    for (int j = 0; j < m_weights.size(); j++)
    {
        // 1. Compute this layer errors
        double sum = 0.;
        for (int k = 0; k < nextLayer->size() - 1; k++)
        {
            sum += nextLayerErrors[k] * nextLayer->weights(k)[j + 1];
        }
        m_errors[j] = m_derivative(m_values[j + 1]) * sum;

        //2. Recompute weights
        auto& weights = m_weights[j];
        for (int l = 0; l < weights.size(); l++)
        {
            weights[l] -= learnRate * m_errors[j] * previousValues[l];
        }
    }
}

void Layer::recomputeWeights(const std::vector<double> &thisLayerErros,
                             const std::vector<double> &previousValues,
                             const double learnRate)
{
    m_errors = thisLayerErros;
#pragma omp parallel for
    for (int j = 0; j < m_weights.size(); j++) // Don't include 1 on the last layer
    {
        auto& weights = m_weights[j];
        for (int l = 0; l < weights.size(); l++)
        {
            weights[l] -= learnRate * m_errors[j] * previousValues[l];
        }
    }
}


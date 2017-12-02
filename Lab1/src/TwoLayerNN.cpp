#include "TwoLayerNN.h"

#include <cmath>
#include <iostream>
#include <iterator>
#include <omp.h>

TwoLayerNN::TwoLayerNN(const int inputSize, const int hiddenSize, const int outputSize)
    : m_inputSize(inputSize)
    , m_hiddenSize(hiddenSize)
    , m_outputSize(outputSize)
    , m_layers(2)
{
    auto superTanh = [](const double sum) {
        if (sum < -20.)
            return -1.;
        else
            if (sum > 20.)
                return 1.;
            else
                return tanh(sum);
        return 0.;
    };

    auto superTanhDerivative = [](const double sum) {
        return (1. - sum) * (1. + sum);
    };

    m_layers[0] = std::make_shared<Layer>(hiddenSize, inputSize, superTanh, superTanhDerivative);
    m_layers[1] = std::make_shared<Layer>(outputSize, hiddenSize);
}

std::vector<double> TwoLayerNN::convertUcharToDouble(const std::vector<unsigned char> &datasetEntry)
{
    std::vector<double> doubleDatasetEntry(datasetEntry.size() + 1);
    doubleDatasetEntry[0] = 1.;
    for (int i = 1; i < datasetEntry.size() + 1; i++)
    {
        doubleDatasetEntry[i] = datasetEntry[i - 1] / 255.;
    }
    return doubleDatasetEntry;
}

void TwoLayerNN::forwardRun(const std::shared_ptr<Layer> &layer,
                            const std::vector<double> &previosValues)
{
    std::vector<double> layerValues(layer->size() - 1);
    for (int j = 0; j < layer->size() - 1; j++)
    {
        auto weights = layer->weights(j);
        double sum = 0.;
        for (int xi = 0; xi < previosValues.size(); xi++)
        {
            sum += weights[xi] * previosValues[xi];
        }
        layerValues[j] = sum;
    }
    layer->activateValues(layerValues);
}

std::vector<double> TwoLayerNN::scaledSoftMax(const std::vector<double> &topLayerValues)
{
    double exponentsSum = 0.;
    double max = *std::max_element(topLayerValues.begin(), topLayerValues.end());
    for (int i = 1; i < topLayerValues.size(); i++) // Don't include 1
    {
        exponentsSum += std::exp(topLayerValues[i] - max);
    }

    std::vector<double> likelihoodValues(topLayerValues.size() - 1);

    for (int i = 0; i < topLayerValues.size() - 1; i++)
    {
        likelihoodValues[i] = exp(topLayerValues[i + 1] - max) / exponentsSum;
    }
    return likelihoodValues;
}

double TwoLayerNN::crossEntropy(const std::vector<std::vector<unsigned char> > &dataset, const std::vector<int> &y)
{
    double currentError = 0.;
    for (int i = 0; i < dataset.size(); i++)
    {
        auto inputs = convertUcharToDouble(dataset[i]);

        auto& lastLayer = m_layers.back();
        auto& firstLayer = m_layers.front();

        forwardRun(firstLayer, inputs);
        forwardRun(lastLayer, firstLayer->values());
        auto likelihoodValues = scaledSoftMax(lastLayer->values());
        currentError -= log(likelihoodValues[y[i]]);
    }
    return currentError / dataset.size();
}

void TwoLayerNN::train(const std::vector<std::vector<unsigned char> > &dataset, const std::vector<int> &y,
                       const int epochCount, const double error, const double learnRate)
{
    m_output.resize(y.size());
    double currentError = error;
    double currentLearnRate =  learnRate;
    for (int epoch = 0; epoch < epochCount; epoch++)
    {
        if (currentError < error)
            break;

        for (int i = 0; i < dataset.size(); i++)
        {
            // 1. Forward Run
            auto inputs = convertUcharToDouble(dataset[i]);

            auto& lastLayer = m_layers.back();
            auto& firstLayer = m_layers.front();

            forwardRun(firstLayer, inputs);
            forwardRun(lastLayer, firstLayer->values());

            auto likelihoodValues = scaledSoftMax(lastLayer->values());
            m_output[i] = static_cast<int> (std::distance(likelihoodValues.begin(),
                                                          std::max_element(likelihoodValues.begin(),
                                                                           likelihoodValues.end())));

            // 2. Back propogation
            auto layerErrors = likelihoodValues;
            layerErrors[y[i]] -= 1;

            auto previousLayerValues = firstLayer->values();
            lastLayer->recomputeWeights(layerErrors, previousLayerValues, currentLearnRate);
            firstLayer->recomputeWeights(lastLayer, inputs, currentLearnRate);
        }

        currentError = crossEntropy(dataset, y);
        std::cout << std::endl << epoch << " Epoch: " << currentError;
    }
}

std::vector<int> TwoLayerNN::predict(const std::vector<std::vector<unsigned char> > &dataset)
{
    std::vector<int> output(dataset.size());
    for (int i = 0; i < dataset.size(); i++)
    {
        auto inputs = convertUcharToDouble(dataset[i]);

        auto& lastLayer = m_layers.back();
        auto& firstLayer = m_layers.front();

        forwardRun(firstLayer, inputs);
        forwardRun(lastLayer, firstLayer->values());

        auto likelihoodValues = scaledSoftMax(lastLayer->values());
        output[i] = static_cast<int> (std::distance(likelihoodValues.begin(),
                                                    std::max_element(likelihoodValues.begin(),
                                                                     likelihoodValues.end())));
    }
    return output;
}

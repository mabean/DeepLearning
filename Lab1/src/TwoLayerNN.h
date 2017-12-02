#ifndef TWOLAYERNN_H
#define TWOLAYERNN_H

#include <array>
#include <memory>
#include <functional>

#include "Layer.h"

class TwoLayerNN
{
public:
    TwoLayerNN(const int inputSize, const int hiddenSize, const int outputSize);

    void train(const std::vector<std::vector<unsigned char> > &dataset, const std::vector<int> &y,
               const int epochCount, const double error, const double learnRate);
    std::vector<int> predict(const std::vector<std::vector<unsigned char> > &dataset);

private:
    std::vector<double> convertUcharToDouble(const std::vector<unsigned char> &datasetEntry);
    void forwardRun(const std::shared_ptr<Layer> &layer,
                    const std::vector<double> &previosValues);
    std::vector<double> scaledSoftMax(const std::vector<double> &topLayerValues);
    double crossEntropy(const std::vector<std::vector<unsigned char> > &dataset, const std::vector<int> &y);

private:
    int m_inputSize;
    int m_hiddenSize;
    int m_outputSize;
    std::vector<std::shared_ptr<Layer> > m_layers;
    std::vector<int> m_output;
};

#endif //TWOLAYERNN_H

#ifndef LAYER_H
#define LAYER_H

#include <functional>
#include <memory>
#include <vector>

class Layer
{
public:
    typedef std::function<double(double)> Activation;
    typedef std::function<double(double)> Derivative;

    Layer(const int size, const int previousLayerSize,
          const Activation &function = identicalFunction,
          const Derivative &derivativeFunc = identicalDerivative);

    static double identicalFunction(const double sum)
    {
        return sum;
    }

    static double identicalDerivative(const double)
    {
        return 1.;
    }

    int size() const;
    std::vector<double> weights(const int nodeIndex) const;
    std::vector<double> errors() const;
    std::vector<double> values() const;
    void activateValues(const std::vector<double> &sums);
    void recomputeWeights(const std::shared_ptr<Layer> &nextLayer,
                          const std::vector<double> &previousValues,
                          const double learnRate);
    void recomputeWeights(const std::vector<double> &thisLayerErros,
                          const std::vector<double> &previousValues,
                          const double learnRate);

protected:
    std::vector<double> m_values;
    std::vector<std::vector<double> > m_weights;
    std::vector<double> m_errors;
    Activation m_function;
    Derivative m_derivative;
};

#endif //LAYER_H

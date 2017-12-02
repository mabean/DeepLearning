#include <iostream>

#include "MnistReader.h"
#include "TwoLayerNN.h"

int main(int argc, char** argv)
{
    std::string pathToXTrain = "C:\\Dev\\University\\Deep Learning\\Lab1\\resource\\x_train.idx3-ubyte";
    std::string pathToyTrain = "C:\\Dev\\University\\Deep Learning\\Lab1\\resource\\y_train.idx1-ubyte";
    std::string pathToXTest = "C:\\Dev\\University\\Deep Learning\\Lab1\\resource\\x_test.idx3-ubyte";
    std::string pathToyTest = "C:\\Dev\\University\\Deep Learning\\Lab1\\resource\\y_test.idx1-ubyte";
    int hiddenLayerSize = 100;
    int epochCount = 25;
    double stopCriteria = 0.02;
    double leartRate = 0.005;

    if (argc >= 5)
    {
        pathToXTrain = argv[1];
        pathToyTrain = argv[2];
        pathToXTest = argv[3];
        pathToyTest = argv[4];
        hiddenLayerSize = atoi(argv[5]);
        epochCount = atoi(argv[6]);
        stopCriteria = atof(argv[7]);
        leartRate = atof(argv[8]);
    }
    else if (argc != 1)
    {
        std::cout << "Usage: BespalovLab1 [PATHTOTRAIN] [PATHTOTRAINLABELS] [PATHTOTEST] [PATHTOTESTLABELS] [HIDDENLAYERSIZE] [EPOCHCOUNT] [STOPCRITERIA] [LEARNRATE]\n";
        return 1;
    }

    int trainSize;
    int imageSize;
    auto X_train = MnistReader::readImages(pathToXTrain, trainSize, imageSize);
    auto y_train = MnistReader::readLabels(pathToyTrain, trainSize);
    int testSize;
    auto X_test = MnistReader::readImages(pathToXTest, testSize, imageSize);
    auto y_test = MnistReader::readLabels(pathToyTest, trainSize);

    auto classes = y_train;
    std::sort(classes.begin(), classes.end());
    auto lastUnique = std::unique(classes.begin(), classes.end());
    classes.erase(lastUnique, classes.end());
    int classesCount = static_cast<int>(classes.size());
    classes.clear();

    auto neuralNetwork = TwoLayerNN(imageSize, hiddenLayerSize, classesCount);

    neuralNetwork.train(X_train, y_train, epochCount, stopCriteria, leartRate);
    auto output = neuralNetwork.predict(X_test);

    int match = 0;
    for (int i = 0; i < testSize; i++)
    {
        if (output[i] == y_test[i])
            match++;
    }
    std::cout << std::endl << "Accuracy reached on test dataset: " << match * 1. / testSize;

    return 0;
}

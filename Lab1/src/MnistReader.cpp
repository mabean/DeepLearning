#include "MnistReader.h"

#include <stdexcept>
#include <fstream>

namespace {
int reverseInt(const int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return static_cast<int>(c1 << 24) + static_cast<int>(c2 << 16) + static_cast<int>(c3 << 8) + c4;
}
}

std::vector<std::vector<unsigned char> > MnistReader::readImages(const std::string &path, int &imagesCount, int &imageSize)
{
    std::ifstream file(path, std::ios::binary);

    if(file.is_open())
    {
        int magicNumber = 0, rowsCount = 0, colsCount = 0;

        file.read((char*)(&magicNumber), sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        if (magicNumber != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char*)(&imagesCount), sizeof(imagesCount));
        imagesCount = reverseInt(imagesCount);
        file.read((char*)(&rowsCount), sizeof(rowsCount));
        rowsCount = reverseInt(rowsCount);
        file.read((char*)(&colsCount), sizeof(colsCount));
        colsCount = reverseInt(colsCount);

        imageSize = rowsCount * colsCount;

        std::vector<std::vector<uchar> > dataset(imagesCount);
        for (int i = 0; i < imagesCount; i++)
        {
            dataset[i].resize(rowsCount * colsCount);
            for(int r = 0; r < rowsCount; r++)
            {
                for(int c = 0; c < colsCount; c++)
                {
                    uchar temp = 0;
                    file.read((char*)&dataset[i][r * rowsCount + c], sizeof(temp));
                }
            }
        }
        return dataset;
    }
    else
    {
        throw std::runtime_error("Cannot open file `" + path + "`!");
    }
}

std::vector<int> MnistReader::readLabels(const std::string path, int &labelsCount)
{
    std::ifstream file(path, std::ios::binary);

    if (file.is_open())
    {
        int magicNumber = 0;
        file.read((char*)(&magicNumber), sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        if (magicNumber != 2049)
            throw std::runtime_error("Invalid MNIST label file!");

        file.read((char*)(&labelsCount), sizeof(labelsCount)), labelsCount = reverseInt(labelsCount);

        std::vector<int> dataset(labelsCount);
        for(int i = 0; i < labelsCount; i++)
        {
            uchar temp = 0;
            file.read((char*)(&temp), 1);
            dataset[i] = static_cast<int> (temp);
        }
        return dataset;
    }
    else
    {
        throw std::runtime_error("Unable to open file `" + path + "`!");
    }
}

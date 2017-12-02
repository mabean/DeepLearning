#ifndef MNISTREADER_H
#define MNISTREADER_H

#include <string>
#include <vector>

class MnistReader {
public:
    typedef unsigned char uchar;

    static std::vector<std::vector<uchar> > readImages(const std::string& path, int& imagesCount, int& imageSize);
    static std::vector<int> readLabels(const std::string path, int& labelsCount);
};


#endif //MNISTREADER_H

#include <iostream>
#include "NeuralNetwork.hpp"
#include <vector>
#include <cstdio>

int main()
{
    std::vector<int> topology = {4,2,4};

    std::vector<sp::Block> Blocklist = {};
    sp::Block block_0(topology);
    sp::Block block_1(topology);
    Blocklist.push_back(block_0);
    Blocklist.push_back(block_1);
    int num_blks = Blocklist.size();


    sp::SimpleNN nn(Blocklist, 1.0f);


    sp::Matrix2D<float> input (4,5);
    //std::fill(input._vals.begin(), input._vals.end(), 1);
    input._vals = {0.0, 0.1, 0.2, 0.3,
                   0.4, 0.4, 0.4, 0.4,
                   0.2, 0.1, 1.2, 1.3,
                   1.2, 1.3, 1.4, 1.5,
                   1.6, 1.7, 1.8, 1.9};


    std::cout << "training start\n";
    nn.feedForward(input, num_blks);

    // test
    std::vector<float> preds = nn.getPredictions();
    std::cout << "training complete\n";
    std::cout << preds[0] <<','<<preds[1] <<','<<preds[2] <<','<<preds[3] <<','<< std::endl;


}
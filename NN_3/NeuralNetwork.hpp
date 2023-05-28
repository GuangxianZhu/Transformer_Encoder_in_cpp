//
// Created by 12192 on 2021/9/25.
//
#pragma once
#include "matrix.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <numeric>


namespace sp {

    //simple activation function
    inline float Sigmoid(float x) {
        return 1.0f / (1 + exp(-x));
    }

    //derivative of activation function
    // x = sigmoid(input);
    inline float DSigmoid(float x) {
        return (x * (1 - x));
    }

    // multi-label activation
    inline Matrix2D<float> Softmax(sp::Matrix2D<float> input){
        sp::Matrix2D<float> exp_input(input._cols, input._rows);
        for(int i=0; i<input._vals.size(); i++){
            exp_input._vals[i] = exp(input._vals[i]);
        }
        float sum_of_exp = std::accumulate(exp_input._vals.begin(), exp_input._vals.end(), decltype(exp_input._vals)::value_type(0));
        sp::Matrix2D<float> result(input._cols, input._rows);
        for(int i=0; i<input._vals.size(); i++){
            result._vals[i] = exp_input._vals[i] / sum_of_exp;
        }
        return result;
    }

    inline Matrix2D<float> D_Softmax(sp::Matrix2D<float> input){ //input:(1row, 3col)
        Matrix2D<float> tensor_1 (input._cols, input._cols);//3*3
        for(int k=0; k<input._cols; k++){
            for(int i=0; i<input._cols; i++){
                tensor_1.at(k,i) = input._vals[k];
            }
        }

        sp::Matrix2D<float> tensor_1_T = tensor_1.transpose();
        tensor_1 = tensor_1.multiplyElements(tensor_1_T);

        Matrix2D<float> tensor_2 (input._cols, input._cols);
        std::fill_n(tensor_2._vals.begin(), input._cols * input._rows, 0);
        for (int i=0; i<input._vals.size(); i++){
            tensor_2.at(i,i) = input._vals[i]; // 对角填充
        }
        Matrix2D<float> tensor_1_nega = tensor_1.negetive();
        Matrix2D<float> result = tensor_2.add(tensor_1_nega); // tsr2-tsr1
        return result; //3*3

    }


    inline sp::Matrix2D<float>Attention(sp::Matrix2D<float> input)
    {
        sp::Matrix2D<float> q(input); //copy input to q
        sp::Matrix2D<float> k(input); //copy input to k
        sp::Matrix2D<float> v(input); //copy input to v

        sp::Matrix2D<float> atten = q.transpose().multiply(k);
        std::cout<< "q*k is ok..."<<std::endl;
        sp::Matrix2D<float> atten_sig = atten.applyFunction(Sigmoid); // square: attention score
        //sp::Matrix2D<float> atten_sig = Softmax(atten); // square: attention score
        std::cout<< "softmax(q*k) is ok..."<<std::endl;
        sp::Matrix2D<float> attention = v.multiply(atten_sig); // same shape as input
        std::cout<< "multi v is ok..."<<std::endl;

        return attention;
    }

    inline sp::Matrix2D<float> Layer_Norm (sp::Matrix2D<float> input, int dim)
    {
        sp::Matrix2D<float> result (input._cols, input._rows);
        float gamma = 1.0;
        float beta = 0.0;
        float epsln = 0.0001;

        for(int row_id=0; row_id<input._rows; row_id++){ // For each row
            float row_sum = 0.0;
            for(int col_id=0; col_id<input._cols; col_id++){
                row_sum += input.at(col_id, row_id); // sum one row elements
            }
            float row_mean = row_sum / input._cols; //mean
            float accum = 0.0;
            for(int col_id=0; col_id<input._cols; col_id++){
                accum += (input.at(col_id, row_id) - row_mean) * (input.at(col_id, row_id) - row_mean);
            }
            float var = accum / input._cols; // variance

            for(int col_id=0; col_id<input._cols; col_id++){
                result.at(col_id, row_id) = (input.at(col_id, row_id)-row_mean)*gamma / sqrt(var+epsln) + beta;
            }
        }
        return result;

    }

    inline sp::Matrix2D<float> celoss(sp::Matrix2D<float> pred, sp::Matrix2D<float> target)
    {
        sp::Matrix2D<float> log_pred(pred._cols, 1);
        for(int i=0; i<pred._vals.size(); i++){ // log the pred
            log_pred._vals[i] = std::log(pred._vals[i]);
        }
        sp::Matrix2D<float> error = log_pred.multiplyElements(target);
        error = error.negetive();
        return error;
    }

// ================================================================================


// ================================================================================
    class Block {
    public:
        std::vector<sp::Matrix2D<float>> _weightMatrices;
        std::vector<sp::Matrix2D<float>> _valueMatrices;
        std::vector<sp::Matrix2D<float>> _biasMatrices;
        std::vector<sp::Matrix2D<float>> _errorMatrices;
        std::vector<sp::Matrix2D<float>> _dOutputMatrices;
        std::vector<sp::Matrix2D<float>> _gradientMatrices;
        std::vector<sp::Matrix2D<float>> _delta_wMatrices;

        std::vector<int> _topology;

    public:
        Block(std::vector<int> topolo) :
                _topology(topolo),
                _weightMatrices({}),
                _valueMatrices({}),
                _biasMatrices({}),
                _errorMatrices({}),
                _dOutputMatrices({}),
                _gradientMatrices({}),
                _delta_wMatrices({}) {
            for (int i = 0; i < topolo.size() - 1; i++) {
                Matrix2D<float> weightMatrix(topolo[i + 1], topolo[i]);
                std::fill_n(weightMatrix._vals.begin(), topolo[i + 1] * topolo[i], 1);//Fill in 1 for testing
//                weightMatrix = weightMatrix.applyFunction([](const float &val) {
//                    return (float) rand() / RAND_MAX;
//                });
                _weightMatrices.push_back(weightMatrix);

                Matrix2D<float> biasMatrix(topolo[i + 1], 1);
                std::fill_n(biasMatrix._vals.begin(), topolo[i + 1] * 1, 1);//Fill in 1 for testing
//                biasMatrix = biasMatrix.applyFunction([](const float &val) {
//                    return (float) rand() / RAND_MAX;
//                });
                _biasMatrices.push_back(biasMatrix);
            }
            _valueMatrices.resize(topolo.size()); //3
            _errorMatrices.resize(topolo.size()); //3
            _dOutputMatrices.resize(topolo.size()); //3
            _gradientMatrices.resize(topolo.size()); //3
            _delta_wMatrices.resize(topolo.size() - 1); //2
        }

    };


    class SimpleNN {
    public:
        std::vector<Block> _Blocklist;
        float _learningRate;

    public:
        SimpleNN(std::vector<Block> Blocklist, float learningRate = 0.1f) :
                _Blocklist(Blocklist),
                _learningRate(learningRate) {

        }

        bool feedForward(sp::Matrix2D<float> values, int num_blk) {
            sp::Matrix2D<float> norm_values (values._cols, values._rows);
            sp::Matrix2D<float> atten_values (values._cols, values._rows);

            for (int blk = 0; blk < num_blk; blk++) { // 0 1 2...
                //values = Attention(values);
                for (int i = 0; i < _Blocklist[blk]._weightMatrices.size(); i++) { //0 1
                    _Blocklist[blk]._valueMatrices[i] = values; //b0v0, b0v1, b1v0, b1v1
                    for (auto value: values._vals)
                        std::cout <<"input, "<<"b"<<blk<<"v"<<i<<": "   <<value << std::endl;

                    norm_values = Layer_Norm(values, 1);
                    for (auto value: norm_values._vals)
                        std::cout <<"norm_values, "<<"b"<<blk<<"v"<<i<<": "   <<value << std::endl;

                    atten_values = Attention(norm_values);
                    for (auto value: atten_values._vals)
                        std::cout <<"atten_values, "<<"b"<<blk<<"v"<<i<<": "   <<value << std::endl;

                    values = values.add(atten_values);

                    values = values.multiply(_Blocklist[blk]._weightMatrices[i]); //input*b0w0, input*b0w1
                    std::cout<< ".....after residual and *weight.........."<<std::endl;
                    for (auto value: values._vals)
                        std::cout <<"input, "<<"b"<<blk<<"v"<<i<<": "   <<value << std::endl;

                    //values = values.add(_Blocklist[blk]._biasMatrices[i]);//+bas0,+bas1
                    values = values.applyFunction(Sigmoid);
                    std::cout<< ".....after sigmoid..........."<<std::endl;
                    for (auto value: values._vals)
                        std::cout <<"input, "<<"b"<<blk<<"v"<<i<<": "   <<value << std::endl;
                    std::cout<< ".....next loop..........."<<std::endl;

                }

                _Blocklist[blk]._valueMatrices[_Blocklist[blk]._weightMatrices.size()] = values;
                for (auto value: values._vals)
                    std::cout <<"values_block_back, "<<"b"<<blk<<"v"<<2<<": "   <<value << std::endl;

            }

            // forward test *************************************
//            for (int blk = 0; blk < num_blk; blk++){
//
//                for (int i = 0; i < _Blocklist[blk]._weightMatrices.size(); i++){ // 1 0
//                    for (auto value: _Blocklist[blk]._weightMatrices[i]._vals)
//                        std::cout <<"b"<<blk<<"w"<<i<<": "   <<value << std::endl;
//                    for (auto value: _Blocklist[blk]._valueMatrices[i]._vals)
//                        std::cout <<"b"<<blk<<"v"<<i<<": "   <<value << std::endl;
//                }
//                for (auto value: _Blocklist[blk]._valueMatrices.back()._vals)
//                    std::cout <<"b"<<blk<<"v"<<_Blocklist[blk]._valueMatrices.size()-1<<": " <<value << std::endl;
//            }

            return true;
//
        }

            std::vector<float> getPredictions() {
                return _Blocklist.back()._valueMatrices.back()._vals;
            }

        };
    }
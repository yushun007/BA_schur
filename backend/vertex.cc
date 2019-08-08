#include "backend/vertex.h"
#include <iostream>

namespace myslam {
namespace backend {

unsigned long global_vertex_id = 0;
//构造函数
Vertex::Vertex(int num_dimension, int local_dimension) {
    parameters_.resize(num_dimension, 1);
    //eigen里对dynamic矩阵(动态大小矩阵)改变其大小可以用resize()函数改变
    local_dimension_ = local_dimension > 0 ? local_dimension : num_dimension;
    //判断顶点自由度是否大于0,如果不大于0,取顶点向量的维度
    //c++判断语句简写a?b:c,先判断a,如果为真则执行b,否则执行c
    id_ = global_vertex_id++;

    std::cout << "Vertex construct num_dimension: " << num_dimension
              << " local_dimension: " << local_dimension << " id_: " << id_ << std::endl;
}
//默认构造函数
Vertex::~Vertex() {}
//返回顶点维度
int Vertex::Dimension() const {
    return parameters_.rows();
}
//返回顶点自由度
int Vertex::LocalDimension() const {
    return local_dimension_;
}
//定义顶点加法
void Vertex::Plus(const VecX &delta) {
    parameters_ += delta;
}

}
}
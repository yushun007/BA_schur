#ifndef MYSLAM_BACKEND_MOTIONVERTEX_H
#define MYSLAM_BACKEND_MOTIONVERTEX_H

#include <memory>
#include "backend/vertex.h"

namespace myslam {
namespace backend {

/**
 * Motion vertex  以加速度计参数的顶点
 * parameters: v, ba, bg 9 DoF 由速度,加速度,重力加速度组成,有9自由度
 * 
 */
class VertexMotion : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexMotion() : Vertex(9) {}

    std::string TypeInfo() const {
        return "VertexMotion";
    }

};

}
}

#endif

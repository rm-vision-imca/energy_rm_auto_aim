#ifndef LEAF_HPP_
#define LEAF_HPP_
#include <opencv2/core.hpp>

namespace rm_auto_aim{
    const std::string LEAF_TYPE_STR[2] = {"INVALID", "VALID"};
    const int RED=0;
    const int BLUE=1;
    enum class LeafType {INVALID=0,VALID=1};
    enum LeafPointType{TOP_LEFT=0,BOTTOM_LEFT,R,BOTTOM_RIGHT,TOP_RIGHT,CENTER_POINT};
    struct Leaf {
        cv::Rect_<float> rect;
        int label;
        float prob;
        LeafType leaf_type;
    /*
    @param [0]top_left
    @param [1]bottom_left
    @param [2]R
    @param [3]bottom_right
    @param [4]top_right
    @param [5]center_point
    */    
        std::vector<cv::Point2f>kpt;
        
    };
    
}
#endif
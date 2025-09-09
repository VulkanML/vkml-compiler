#pragma once
#include <vector>
#include <string>
#include "OperationFunction.h"

class ProgramSegment {
public:
    ProgramSegment(const std::string& segmentId, const std::vector<OperationFunction>& ops, int deviceId);
    std::string getSegmentId() const;
    const std::vector<OperationFunction>& getOps() const;
    int getDeviceId() const;
private:
    std::string segmentId_;
    std::vector<OperationFunction> ops_;
    int deviceId_;
};

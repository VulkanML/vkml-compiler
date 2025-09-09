#include "ProgramSegment.h"

ProgramSegment::ProgramSegment(const std::string& segmentId, const std::vector<OperationFunction>& ops, int deviceId)
    : segmentId_(segmentId), ops_(ops), deviceId_(deviceId) {}

std::string ProgramSegment::getSegmentId() const { return segmentId_; }
const std::vector<OperationFunction>& ProgramSegment::getOps() const { return ops_; }
int ProgramSegment::getDeviceId() const { return deviceId_; }

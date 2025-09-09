#include "CompilationConfiguration.h"

CompilationConfiguration::CompilationConfiguration(const std::vector<int>& targetDevices, const std::vector<std::string>& debugFlags, int optimizationLevel)
    : targetDevices_(targetDevices), debugFlags_(debugFlags), optimizationLevel_(optimizationLevel) {}

const std::vector<int>& CompilationConfiguration::getTargetDevices() const { return targetDevices_; }
const std::vector<std::string>& CompilationConfiguration::getDebugFlags() const { return debugFlags_; }
int CompilationConfiguration::getOptimizationLevel() const { return optimizationLevel_; }

#pragma once
#include <vector>
#include <string>

class CompilationConfiguration {
public:
    CompilationConfiguration(const std::vector<int>& targetDevices, const std::vector<std::string>& debugFlags, int optimizationLevel);
    const std::vector<int>& getTargetDevices() const;
    const std::vector<std::string>& getDebugFlags() const;
    int getOptimizationLevel() const;
private:
    std::vector<int> targetDevices_;
    std::vector<std::string> debugFlags_;
    int optimizationLevel_;
};

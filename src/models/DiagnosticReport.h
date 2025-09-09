#pragma once
#include <vector>
#include <string>
#include <map>

class DiagnosticReport {
public:
    DiagnosticReport(const std::vector<std::string>& errors, const std::vector<std::string>& warnings, const std::map<std::string, int>& sourceMap);
    const std::vector<std::string>& getErrors() const;
    const std::vector<std::string>& getWarnings() const;
    const std::map<std::string, int>& getSourceMap() const;
private:
    std::vector<std::string> errors_;
    std::vector<std::string> warnings_;
    std::map<std::string, int> sourceMap_;
};

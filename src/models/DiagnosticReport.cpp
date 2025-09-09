#include "DiagnosticReport.h"

DiagnosticReport::DiagnosticReport(const std::vector<std::string>& errors, const std::vector<std::string>& warnings, const std::map<std::string, int>& sourceMap)
    : errors_(errors), warnings_(warnings), sourceMap_(sourceMap) {}

const std::vector<std::string>& DiagnosticReport::getErrors() const { return errors_; }
const std::vector<std::string>& DiagnosticReport::getWarnings() const { return warnings_; }
const std::map<std::string, int>& DiagnosticReport::getSourceMap() const { return sourceMap_; }

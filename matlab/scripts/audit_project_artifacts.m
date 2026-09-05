function report = audit_project_artifacts()
% Script wrapper redirecting to matlab/audit/audit_project_artifacts.m
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'audit'));
    report = audit_project_artifacts();
end

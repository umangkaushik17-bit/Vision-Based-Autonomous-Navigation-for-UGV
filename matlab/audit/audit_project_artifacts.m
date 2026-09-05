function report = audit_project_artifacts()
% AUDIT_PROJECT_ARTIFACTS Scans the repository for usable Phase 1-6 artifacts.
% Inspects configs, data, outputs, ai, and models without hardcoding paths.

    fprintf('==============================================================================\n');
    fprintf('          SIH26126 PHASE 7: REPOSITORY ARTIFACT AUDIT\n');
    fprintf('==============================================================================\n\n');

    % Root search directories relative to repository root
    search_dirs = {'configs', 'data', 'outputs', 'ai', 'models', 'matlab/data'};
    target_exts = {'.yaml', '.json', '.mat', '.npz', '.csv', '.png', '.jpg', '.pt', '.onnx'};

    artifacts = struct('name', {}, 'path', {}, 'type', {}, 'bytes', {}, 'phase', {}, 'matlab_use', {}, 'readable', {});
    counter = 0;

    for d_idx = 1:length(search_dirs)
        dir_name = search_dirs{d_idx};
        if ~exist(dir_name, 'dir')
            continue;
        end

        files = dir(fullfile(dir_name, '**', '*.*'));
        for f_idx = 1:length(files)
            f = files(f_idx);
            if f.isdir
                continue;
            end

            [~, fname, ext] = fileparts(f.name);
            ext = lower(ext);

            if any(strcmp(target_exts, ext))
                counter = counter + 1;
                full_p = fullfile(f.folder, f.name);

                % Map to relevant phase
                rel_phase = 'Phase 0-1';
                matlab_use = 'Inspection';

                if contains(full_p, 'config')
                    rel_phase = 'System / Config';
                    matlab_use = 'Configuration reference';
                elseif contains(full_p, 'camera') || contains(full_p, 'intrinsics')
                    rel_phase = 'Phase 3 (Camera)';
                    matlab_use = 'Calibration validation';
                elseif contains(full_p, 'costmap')
                    rel_phase = 'Phase 5 (Costmap)';
                    matlab_use = 'Costmap cross-validation';
                elseif contains(full_p, 'planning') || contains(full_p, 'trajectory') || contains(full_p, 'astar')
                    rel_phase = 'Phase 6 (Planning)';
                    matlab_use = 'Trajectory / path analysis';
                elseif contains(full_p, 'slam')
                    rel_phase = 'Phase 7-8 (SLAM)';
                    matlab_use = 'ATE/RPE evaluation';
                elseif contains(full_p, 'data')
                    rel_phase = 'Phase 2 (Datasets)';
                    matlab_use = 'Image / ground truth import';
                end

                % Test read capability
                readable = false;
                if strcmp(ext, '.mat')
                    try
                        whos('-file', full_p);
                        readable = true;
                    catch
                        readable = false;
                    end
                elseif any(strcmp(ext, {'.json', '.yaml', '.csv', '.png', '.jpg'}))
                    fid = fopen(full_p, 'r');
                    if fid ~= -1
                        readable = true;
                        fclose(fid);
                    end
                else
                    readable = exist(full_p, 'file') == 2;
                end

                artifacts(counter).name = f.name;
                artifacts(counter).path = full_p;
                artifacts(counter).type = ext;
                artifacts(counter).bytes = f.bytes;
                artifacts(counter).phase = rel_phase;
                artifacts(counter).matlab_use = matlab_use;
                artifacts(counter).readable = readable;
            end
        end
    end

    fprintf('Audited %d candidate artifacts across repository:\n', length(artifacts));
    fprintf('----------------------------------------------------------------------------------------------------\n');
    fprintf('%-30s | %-12s | %-16s | %-8s | %-20s\n', 'Artifact File', 'Type', 'Phase', 'Readable', 'MATLAB Usage');
    fprintf('----------------------------------------------------------------------------------------------------\n');

    % Print first 25 items cleanly
    num_to_print = min(25, length(artifacts));
    for i = 1:num_to_print
        r_str = 'NO';
        if artifacts(i).readable
            r_str = 'YES';
        end
        fprintf('%-30s | %-12s | %-16s | %-8s | %-20s\n', ...
            artifacts(i).name(1:min(30, length(artifacts(i).name))), ...
            artifacts(i).type, ...
            artifacts(i).phase, ...
            r_str, ...
            artifacts(i).matlab_use);
    end

    if length(artifacts) > 25
        fprintf('... and %d more files audited.\n', length(artifacts) - 25);
    end
    fprintf('----------------------------------------------------------------------------------------------------\n');

    report = struct();
    report.total_artifacts = length(artifacts);
    report.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    report.artifacts = artifacts;

    % Save audit summary to results/reports
    if ~exist('matlab/results/reports', 'dir')
        mkdir('matlab/results/reports');
    end
    save('matlab/results/reports/artifact_audit_report.mat', 'report');
    fprintf('[OK] Artifact audit completed. Saved to matlab/results/reports/artifact_audit_report.mat\n\n');
end

function env = check_matlab_environment()
% CHECK_MATLAB_ENVIRONMENT Check MATLAB version, host platform, and installed toolboxes.
% Gracefully reports missing optional toolboxes without raising fatal errors.

    fprintf('==============================================================================\n');
    fprintf('           SIH26126 PHASE 7: MATLAB ENVIRONMENT AUDIT\n');
    fprintf('==============================================================================\n');

    v = ver;
    env = struct();
    env.version = version;
    env.os = computer;
    env.is_windows = ispc;
    env.is_linux = isunix && ~ismac;
    env.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    fprintf('  MATLAB Version : %s\n', env.version);
    fprintf('  Platform / OS  : %s\n', env.os);
    fprintf('  Timestamp      : %s\n', env.timestamp);

    % Inspect installed toolboxes
    installed_toolboxes = {v.Name};
    env.installed_toolboxes = installed_toolboxes;

    % Optional toolboxes of interest
    optional_toolboxes = {
        'Computer Vision Toolbox',
        'Image Processing Toolbox',
        'Navigation Toolbox',
        'Robotics System Toolbox',
        'Optimization Toolbox',
        'Statistics and Machine Learning Toolbox'
    };

    fprintf('\n--- Optional Toolbox Availability ---\n');
    env.optional_status = struct();
    for i = 1:length(optional_toolboxes)
        tb_name = optional_toolboxes{i};
        has_tb = any(strcmp(installed_toolboxes, tb_name));
        field_key = lower(regexprep(tb_name, '[^a-zA-Z0-9]', '_'));
        env.optional_status.(field_key) = has_tb;

        if has_tb
            fprintf('  [AVAILABLE]     : %s\n', tb_name);
        else
            fprintf('  [NOT INSTALLED] : %s (Graceful Pure-MATLAB Fallback Active)\n', tb_name);
        end
    end

    fprintf('\n[OK] Environment check completed. All core validation runs in pure MATLAB.\n\n');
end

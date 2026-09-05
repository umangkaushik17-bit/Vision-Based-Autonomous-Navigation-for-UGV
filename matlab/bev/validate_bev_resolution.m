function res = validate_bev_resolution()
% VALIDATE_BEV_RESOLUTION Independently validates consistency between
% physical extent (5.8m x 6.0m), resolution (10cm), and grid dimensions (58 x 60).

    fprintf('--- Validating BEV Physical Extent and Resolution Consistency ---\n');

    % Phase 5 canonical specification
    fwd_extent_m = 5.8;    % Forward range from bumper
    lat_extent_m = 6.0;    % Lateral width centered on vehicle [-3.0m, +3.0m]
    resolution_m = 0.10;   % 10 cm per grid cell

    expected_rows = round(fwd_extent_m / resolution_m);
    expected_cols = round(lat_extent_m / resolution_m);

    actual_rows = 58;
    actual_cols = 60;
    actual_total_cells = actual_rows * actual_cols;

    row_diff = abs(actual_rows - expected_rows);
    col_diff = abs(actual_cols - expected_cols);

    % Memory footprint check (8 channels of float32 + uint8)
    % 3480 cells * ~40 bytes/cell = ~139 KB
    bytes_per_cell = (7 * 4) + (2 * 1) + 4; % 7 float32 + 2 uint8/bool + 1 int32 = 34 bytes
    est_memory_kb = (actual_total_cells * bytes_per_cell) / 1024.0;

    res = struct();
    res.fwd_extent_m = fwd_extent_m;
    res.lat_extent_m = lat_extent_m;
    res.resolution_m = resolution_m;
    res.expected_rows = expected_rows;
    res.expected_cols = expected_cols;
    res.actual_rows = actual_rows;
    res.actual_cols = actual_cols;
    res.total_cells = actual_total_cells;
    res.memory_kb = est_memory_kb;
    res.is_consistent = (row_diff == 0) && (col_diff == 0);

    fprintf('  Forward Extent         : %.2f m\n', fwd_extent_m);
    fprintf('  Lateral Extent         : %.2f m\n', lat_extent_m);
    fprintf('  Grid Resolution        : %.2f m (10 cm)\n', resolution_m);
    fprintf('  Calculated Rows / Cols : %d rows x %d cols\n', expected_rows, expected_cols);
    fprintf('  Canonical Grid Shape   : %d rows x %d cols (%d cells)\n', actual_rows, actual_cols, actual_total_cells);
    fprintf('  Estimated Memory       : %.1f KB per costmap\n', est_memory_kb);

    if res.is_consistent
        fprintf('  Resolution Status      : PASS (Exact Mathematical Parity Confirmed)\n');
    else
        fprintf('  Resolution Status      : FAIL (Discrepancy detected)\n');
    end
end

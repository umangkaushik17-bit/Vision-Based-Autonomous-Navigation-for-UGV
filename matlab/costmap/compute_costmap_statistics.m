function stats = compute_costmap_statistics(cm_struct, scenario_name)
% COMPUTE_COSTMAP_STATISTICS Computes distribution, areas, and statistical metrics for a BEV costmap.
% Note: High blocked area is reported descriptively, NOT as inherently superior.

    if nargin < 2
        scenario_name = 'Costmap';
    end

    cost = double(cm_struct.cost);
    obs = logical(cm_struct.obstacle);
    unk = logical(cm_struct.unknown);
    haz = double(cm_struct.hazard) > 0.5;

    N_total = numel(cost);
    res_m = double(cm_struct.resolution_m);
    cell_area_m2 = res_m * res_m;
    total_area_m2 = N_total * cell_area_m2;

    % Thresholds:
    % Safe: C < 0.35 and not obstacle and not unknown
    % Caution: 0.35 <= C < 0.70 and not obstacle and not unknown
    % Lethal: C >= 0.70 or obstacle
    % Unknown: unknown flag set
    safe_mask = (cost < 0.35) & (~obs) & (~unk);
    caution_mask = (cost >= 0.35) & (cost < 0.70) & (~obs) & (~unk);
    lethal_mask = (cost >= 0.70) | obs;
    unk_mask = unk;

    pct_safe = (sum(safe_mask(:)) / N_total) * 100.0;
    pct_caution = (sum(caution_mask(:)) / N_total) * 100.0;
    pct_lethal = (sum(lethal_mask(:)) / N_total) * 100.0;
    pct_unknown = (sum(unk_mask(:)) / N_total) * 100.0;

    stats = struct();
    stats.scenario_name = scenario_name;
    stats.total_cells = N_total;
    stats.total_area_m2 = total_area_m2;
    stats.pct_safe = pct_safe;
    stats.pct_caution = pct_caution;
    stats.pct_lethal = pct_lethal;
    stats.pct_unknown = pct_unknown;
    stats.safe_area_m2 = sum(safe_mask(:)) * cell_area_m2;
    stats.lethal_area_m2 = sum(lethal_mask(:)) * cell_area_m2;
    stats.hazard_area_m2 = sum(haz(:)) * cell_area_m2;
    stats.unknown_area_m2 = sum(unk_mask(:)) * cell_area_m2;
    stats.mean_cost = mean(cost(:));
    stats.median_cost = median(cost(:));
    stats.max_cost = max(cost(:));

    % Count connected components of blocked/lethal regions using pure MATLAB 4-connected floodfill
    stats.num_blocked_components = count_connected_components(lethal_mask);

    fprintf('--- Statistics for: %s ---\n', scenario_name);
    fprintf('  Total Grid Area   : %.2f m^2 (%d cells)\n', total_area_m2, N_total);
    fprintf('  Safe Area (C<0.35): %.2f m^2 (%5.1f%%)\n', stats.safe_area_m2, pct_safe);
    fprintf('  Caution Area      : %5.1f%%\n', pct_caution);
    fprintf('  Lethal Area       : %.2f m^2 (%5.1f%%)\n', stats.lethal_area_m2, pct_lethal);
    fprintf('  Unknown Area      : %.2f m^2 (%5.1f%%)\n', stats.unknown_area_m2, pct_unknown);
    fprintf('  Mean Cost         : %.3f (Median=%.3f, Max=%.2f)\n', stats.mean_cost, stats.median_cost, stats.max_cost);
    fprintf('  Blocked Clusters  : %d connected regions\n', stats.num_blocked_components);
end


function num_clusters = count_connected_components(binary_mask)
    % Pure-MATLAB 4-connected flood fill without requiring Image Processing Toolbox
    [rows, cols] = size(binary_mask);
    visited = false(rows, cols);
    num_clusters = 0;

    for r = 1:rows
        for c = 1:cols
            if binary_mask(r, c) && ~visited(r, c)
                num_clusters = num_clusters + 1;
                % BFS queue
                queue = [r, c];
                visited(r, c) = true;
                head = 1;
                while head <= size(queue, 1)
                    curr = queue(head, :);
                    head = head + 1;
                    cr = curr(1);
                    cc = curr(2);

                    neighbors = [cr-1, cc; cr+1, cc; cr, cc-1; cr, cc+1];
                    for k = 1:4
                        nr = neighbors(k, 1);
                        nc = neighbors(k, 2);
                        if nr >= 1 && nr <= rows && nc >= 1 && nc <= cols
                            if binary_mask(nr, nc) && ~visited(nr, nc)
                                visited(nr, nc) = true;
                                queue(end+1, :) = [nr, nc];
                            end
                        end
                    end
                end
            end
        end
    end
end

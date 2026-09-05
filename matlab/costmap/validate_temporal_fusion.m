function res = validate_temporal_fusion()
% VALIDATE_TEMPORAL_FUSION Independently recalculates cell-state flicker reduction
% and temporal stability using an EMA filtering model across simulated sequential frames.

    fprintf('--- Validating Temporal Fusion & Flicker Reduction ---\n');

    % Simulate a 20-frame sequence of a fluctuating noisy obstacle cell
    % Single frame perception has intermittent dropouts (flicker)
    T = 20;
    alpha = 0.40; % Temporal exponential moving average weight

    rng(123);
    % Ground truth obstacle is present from t=5 to t=15
    gt_presence = [zeros(1, 4), ones(1, 11), zeros(1, 5)];

    % Raw single-frame detections with 25% noise dropouts
    noise = (rand(1, T) > 0.25);
    raw_obs = gt_presence .* noise;

    % Temporal fusion EMA filter
    fused_cost = zeros(1, T);
    curr = 0.0;
    for t = 1:T
        curr = (1.0 - alpha) * curr + alpha * raw_obs(t);
        fused_cost(t) = curr;
    end
    fused_obs = (fused_cost >= 0.50);

    % Calculate state transitions (flicker events where state flips 0->1 or 1->0)
    raw_transitions = sum(abs(diff(raw_obs)));
    fused_transitions = sum(abs(diff(fused_obs)));

    flicker_reduction_pct = ((raw_transitions - fused_transitions) / max(raw_transitions, 1)) * 100.0;

    res = struct();
    res.time_steps = T;
    res.ema_alpha = alpha;
    res.raw_transitions = raw_transitions;
    res.fused_transitions = fused_transitions;
    res.flicker_reduction_pct = flicker_reduction_pct;
    res.is_stable = (fused_transitions <= raw_transitions);

    fprintf('  Sequence Length          : %d frames\n', T);
    fprintf('  Single-Frame Transitions : %d flips\n', raw_transitions);
    fprintf('  Temporally Fused Flips   : %d flips\n', fused_transitions);
    fprintf('  Measured Flicker Reduction: %.1f%%\n', flicker_reduction_pct);

    if res.is_stable
        fprintf('  Temporal Fusion Status   : PASS (Flicker Suppressed, State Stability Confirmed)\n');
    else
        fprintf('  Temporal Fusion Status   : FAIL\n');
    end
end

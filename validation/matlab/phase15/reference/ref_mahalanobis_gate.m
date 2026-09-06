function [d2, d_mahal, is_accepted] = ref_mahalanobis_gate(y, S, chi2_thresh)
% REF_MAHALANOBIS_GATE Independent Mahalanobis gating evaluation in MATLAB.
%
% Inputs:
%   y           - 2x1 innovation residual vector
%   S           - 2x2 innovation covariance matrix
%   chi2_thresh - Chi-square gating boundary (e.g. 9.21)
%
% Outputs:
%   d2          - Mahalanobis distance squared d^2 = y^T * S^-1 * y
%   d_mahal     - Mahalanobis distance sqrt(d^2)
%   is_accepted - True if d2 <= chi2_thresh, false otherwise

    if nargin < 3 || isempty(chi2_thresh)
        chi2_thresh = 9.21;
    end

    y_col = y(:);
    
    % Analytical 2x2 matrix inversion
    det_S = S(1,1) * S(2,2) - S(1,2) * S(2,1);
    if abs(det_S) < 1e-14
        S_inv = pinv(S);
    else
        inv_det = 1.0 / det_S;
        S_inv = [ S(2,2) * inv_det, -S(1,2) * inv_det;
                 -S(2,1) * inv_det,  S(1,1) * inv_det];
    end

    d2 = double(y_col' * S_inv * y_col);
    d_mahal = sqrt(max(0.0, d2));
    is_accepted = (d2 <= chi2_thresh);
end

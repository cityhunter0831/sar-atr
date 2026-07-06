function w = taylorwin(N, nbar, sll)
    % Custom implementation of Taylor window to avoid Signal Processing Toolbox dependency
    if nargin < 3, sll = -30; end
    if nargin < 2, nbar = 4; end
    
    sll = -abs(sll); % Ensure sll is negative
    
    % Calculate A parameter
    A = acosh(10^(-sll/20)) / pi;
    
    % Calculate sigma^2 parameter
    sigma2 = nbar^2 / (A^2 + (nbar - 0.5)^2);
    
    % Calculate the coefficients Fm
    Fm = zeros(nbar - 1, 1);
    for m = 1:(nbar - 1)
        num = 1;
        for j = 1:(nbar - 1)
            num = num * (1 - m^2 / (sigma2 * (A^2 + (j - 0.5)^2)));
        end
        
        den = 1;
        for j = 1:(nbar - 1)
            if j ~= m
                den = den * (1 - m^2 / j^2);
            end
        end
        
        Fm(m) = ((-1)^(m+1) * num) / (2 * den);
    end
    
    % Compute the window coefficients
    w = zeros(N, 1);
    Nc = (N - 1) / 2;
    for n = 1:N
        val = 1;
        for m = 1:(nbar - 1)
            val = val + 2 * Fm(m) * cos(2 * pi * m * (n - 1 - Nc) / N);
        end
        w(n) = val;
    end
end

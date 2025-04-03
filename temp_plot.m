k = 50;         
rho = 7800;     
c_p = 500;       
alpha = 1.282051e-5; 
a = 0.02;        
T_inf = 50;      
Ti = 200;        
h = 100;        
N = 50;         
x = linspace(0, a, 1000); 
time_intervals = linspace(5, 50, 10); 

lambda_n = zeros(1, N);
for n = 1:N
    fun = @(lambda) tan(lambda * a) - (lambda * k / h);
    lambda_n(n) = fzero(fun, (n * pi) / (2 * a));
end

figure; hold on;
colors = lines(length(time_intervals));
for j = 1:length(time_intervals)
    t = time_intervals(j);
    T = T_inf * ones(size(x)); % Initialize with T_inf
    for n = 1:N
        coeff = (2/a) * integral(@(x) (Ti - T_inf) .* sin(lambda_n(n) * x), 0, a);
        T = T + coeff * exp(-lambda_n(n)^2 * alpha * t) .* cos(lambda_n(n) * x);
    end
    plot(x, T, 'Color', colors(j,:), 'LineWidth', 1.5, 'DisplayName', ['t = ', num2str(t), ' sec']);
end

xlabel('x (m)');
ylabel('T(x,t) (C)');
title('Temperature Distribution in the Steel Plate Over Time');
legend show;
grid on;

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
    q_dot = k * T_inf * ones(size(x)); % Initialize with T_inf
    for n = 1:N
        coeff = (2/a) * integral(@(x) (Ti - T_inf) .* sin(lambda_n(n) * x), 0, a);
        term = - k * coeff * lambda_n(n) * exp(-lambda_n(n)^2 * alpha * t) .* sin(lambda_n(n) * x);
        q_dot = q_dot - term;
    end
    plot(x, q_dot, 'Color', colors(j,:), 'LineWidth', 1.5, 'DisplayName', ['t = ', num2str(t), ' sec']);
end

xlabel('x (m)');
ylabel('\dot{q}(W/m^2)');
title('Heat flux Distribution in the Steel Plate Over Time');
legend show;
grid on;

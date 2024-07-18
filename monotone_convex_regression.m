function [p,x, aux_out] = monotone_convex_regression(degree,features,response,monotone_profile,convex_sign)
%% Description 
% Outputs
%   p := decision polynomial
%   x := argument of p
% Inputs
%   degree := the degree  of the decision polynomial 
%   features := feature variable data used for training
%   response := response variable used for training
%   monotone_profile := the vector of 0,1,-1, describing the monotonicity
%                       relationship between each feature and the response
%   convex_sign := 1 if convex, -1 if concave

%% Clear up the environment
% This is an important step for improving performance
yalmip('clear');
%% PROBLEM SETUP: Define the box
% find the superior and inferior bounds 
% (currently we infer the bounds based on the full datasets, 
% in actual applications rescalling is necessary due to 
% numerical problems
tol = 0.001;
inf_domain = min(features) - tol;
sup_domain = max(features) + tol;

%% PROBLEM SETUP: Define the parameters and decision variables

% N-number of data; k - number of features
[N, k] = size(features); 
% Create a 1xk vector of symbolic variables x, denoted as x = [x1, x2, ...,xk]
x=sdpvar(1,k);

% Define the main polynomial to be learned
% p is the polynomial
% c is the array of the cofficients of the polynomial p
% v is the array of monomials
[p,c,v] = polynomial(x,degree);
% e.g., x = [x1, x2], degree = 2
% polynomial: p = c1*x1^2 + c2*x1*x2 + c3*x2^2 + c4*x1 + c5*x2 + c6
% cofficients of the polynomial p: c = [c1; c2; c3; c4; c5; c6]
% monomials in polynomial p: v = [x1^2; x1*x2; x2^2; x1; x2; 1]

%% PROBLEM SETUP: Write the objective
% currently computing the objective is the biggest computational bottleneck
% I tried to vectorized versions, but they lead to errors

monom_bulk = bulkeval(v,x,features'); % <- evaluates all monomials for all values of the features
%{ e.g., features = [1 2;
            2 3;
            3 4;
            4 5;
            5 6;
            6 7;
            7 8;
            8 9;
            9 10;
            10 11]; 10 data 2 features %}
%{ monom_bulk =      1     4     9    16    25    36    49    64    81   100
     2     6    12    20    30    42    56    72    90   110
     4     9    16    25    36    49    64    81   100   121
     1     2     3     4     5     6     7     8     9    10
     2     3     4     5     6     7     8     9    10    11
     1     1     1     1     1     1     1     1     1     1
  where monom_bulk is a matrix of 6 (number of monomials in p, i.e., number of v) * 10 (number of data) %}

peval_bulk = c'*monom_bulk;% <- computes the value of the polynomial given the value of
                           %    the argument from future, as a function of
                           %    polynomial coefficients c
%{  c' is the transpose of c, according to the above example:
    c'=[c1, c2, c3, c4, c5, c6]
    c'*monom_bulk = [ (c1*1 + c2*2 + c3*4 + c4*1 + c5*2 + c6*1),
               (c1*4 + c2*6 + c3*9 + c4*2 + c5*3 + c6*1),
               ...
               (c1*100 + c2*110 + c3*121 + c4*10 + c5*11 + c6*1) ]
    where each element represents the value of the polynomial p for each data %}

diff_bulk = peval_bulk - response'; % <- computes the difference between
                                    %    the function value at the feature
                                    %    input and the response, (as a
                                    %    function of c)
h = norm(diff_bulk); % <- h is the minimization objective, the sum of 
                          %    squared errors
                     % <- norm compute the L2 Norm

%% PROBLEM SETUP: Define the decision variables used in the constraints

%% Monotone
% Create the monomials of the helper polynomials used in the 
% MONOTONE constraints
m_monomials = monolist(x, degree-2);

% Define the coefficients of the matrix of helper polynomials
% for the MONOTONE constraints
m_coef_help = sdpvar(k*k, length(m_monomials));

% Create the matrix of helper polynomials 
m_Q_help = m_coef_help*m_monomials;
m_Q_help = reshape(m_Q_help, k, k, []);

%% Convex
% Create helper free variable for the CONVEX constraints
y=sdpvar(1,k);

% Create the monomials of the helper polynomials used in the constraints
mono_degree = cat(2, repelem(degree-2, k), repelem(2, k));
% (the max degree associated with the helper variable is 2)
c_monomials = monolist([x y], mono_degree);

% Define the coefficients of the array of helper polynomials
% for the CONVEX constraint
c_coef_help = sdpvar(k, length(c_monomials));

% Create an array of helper polynomials 
c_Q_help = c_coef_help*c_monomials;

%% PROBLEM SETUP: Write the constraints
t0 = tic();
F = [sos(m_Q_help), sos(c_Q_help)];
% Add monotonicity constraints
F = F+[sos(transpose(jacobian(p,x)).*monotone_profile - m_Q_help*transpose((x-inf_domain).*(sup_domain-x)))];
% Add convexity constraints
F = F+[sos(y*hessian(p,x)*transpose(y).*convex_sign-(x-inf_domain).*(sup_domain-x)*c_Q_help)];

%% SOS OPTIMIZATION: Fit the desired polynomial
options = sdpsettings('verbose',0, 'solver', 'mosek');
% The coefficients are the decision variables, putting them all in an array
all_coef = [c;reshape(c_coef_help, k*length(c_monomials),1,[]);reshape(m_coef_help, k*k*length(m_monomials),1,[])];
setup_time = toc(t0);
msg = "Setup time: " + setup_time + " seconds.";
disp(msg);
t1 = tic();
[sol,m,B,residual]=solvesos(F, h, options, all_coef);
optimization_time = toc(t1);
msg = "Optimization runtime: " + optimization_time + " seconds.";
disp(msg);


%% Display message
msg = "Monotone-convex regression for polynomial of degree "+degree+" complete.";
aux_out = struct('setup_time', setup_time, 'optimization_time',...
    optimization_time, 'solver_time', sol.('solvertime'), 'train_rmse', sqrt(value(h)^2/N));
end

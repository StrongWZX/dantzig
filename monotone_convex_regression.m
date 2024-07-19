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
% min or max default operation on rows, e.g., features is N*k matrix, and inf_domain is 1*k matrix, 

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
            10 11]; 10 data 2 features
%}
%{ monom_bulk =      1     4     9    16    25    36    49    64    81   100
     2     6    12    20    30    42    56    72    90   110
     4     9    16    25    36    49    64    81   100   121
     1     2     3     4     5     6     7     8     9    10
     2     3     4     5     6     7     8     9    10    11
     1     1     1     1     1     1     1     1     1     1
  where monom_bulk is a matrix of 6 (number of monomials in p, i.e., number of v) * 10 (number of data) 
%}

peval_bulk = c'*monom_bulk;% <- computes the value of the polynomial given the value of
                           %    the argument from future, as a function of
                           %    polynomial coefficients c
%{  c' is the transpose of c, according to the above example:
    c'=[c1, c2, c3, c4, c5, c6]
    c'*monom_bulk = [ (c1*1 + c2*2 + c3*4 + c4*1 + c5*2 + c6*1),
               (c1*4 + c2*6 + c3*9 + c4*2 + c5*3 + c6*1),
               ...
               (c1*100 + c2*110 + c3*121 + c4*10 + c5*11 + c6*1) ]
    where each element represents the value of the polynomial p for each data
%}

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
m_monomials = monolist(x, degree-2); % <- as the following (x-inf_domain).*(sup_domain-x) will prodive additonal 2 degree
%{ m_monomials will contain all monomials of x with a degree of at most (degree-2)
e.g., x = [x1, x2]; degree-2 = 2
m_monomials = [1; x1; x2; x1^2; x1*x2; x2^2]
%}

% Define the coefficients of the matrix of helper polynomials
% for the MONOTONE constraints
m_coef_help = sdpvar(k*k, length(m_monomials));
%{ Generate a symbolic variable matrix with k*k (the first k is the elementer number of following jacobian vector) rows and length(m_monomials) columns
                                                the second k is the number of features
e.g., k*k=4, length(m_monomials)=6;
m_coef_help = [ m11, m12, m13, m14, m15, m16;
                m21, m22, m23, m24, m55, m26;
                m31, m32, m33, m34, m35, m36;
                m41, m42, m43, m44, m45, m46 ]
%}

% Create the matrix of helper polynomials 
m_Q_help = m_coef_help*m_monomials;
%{ according to the above example:
m_Q_help = [ m11*1 + m12*x1 + m13*x2 + m14*x1^2 + m15*x1*x2 + m16*x2^2;
  m21*1 + m22*x1 + m23*x2 + m24*x1^2 + m25*x1*x2 + m26*x2^2;
  m31*1 + m32*x1 + m33*x2 + m34*x1^2 + m35*x1*x2 + m36*x2^2;
  m41*1 + m42*x1 + m43*x2 + m44*x1^2 + m45*x1*x2 + m46*x2^2;]
%}
m_Q_help = reshape(m_Q_help, k, k, []); % <- The original matrix is reshaped into a k*k*n matrix, where n is automatically determined based on the original matrix size and k
%{ according to the above example:
m_Q_help = [ m11*1 + m12*x1 + m13*x2 + m14*x1^2 + m15*x1*x2 + m16*x2^2; m31*1 + m32*x1 + m33*x2 + m34*x1^2 + m35*x1*x2 + m36*x2^2;
             m21*1 + m22*x1 + m23*x2 + m24*x1^2 + m25*x1*x2 + m26*x2^2; m41*1 + m42*x1 + m43*x2 + m44*x1^2 + m45*x1*x2 + m46*x2^2;]
%}

%% Convex
% Create helper free variable for the CONVEX constraints
y=sdpvar(1,k);
% y = [y1, y2, ...,yk]

% Create the monomials of the helper polynomials used in the constraints
mono_degree = cat(2, repelem(degree-2, k), repelem(2, k));
%{ 1. repelem(degree-2, k) generates a vector with k elements, each of which has the value degree-2
e.g., degree-2=2, k=3 -> repelem(degree-2, k) = [2, 2 ,2]
2. cat(dim, A, B) joins the arrays A and B along the dimension dim
cat(2, repelem(degree-2, k), repelem(2, k)) -> [2, 2, 2, 2, 2, 2]
%}
% (the max degree associated with the helper variable is 2)
c_monomials = monolist([x y], mono_degree);
%{monolist([x y], mono_degree) generates all possible monomials, with the highest degree of variables [x,y] specified by the corresponding values in mono_degree
e.g., x=[x1,x2], y=[y1,y2] ->[x,y]=[x1,x2,y1,y2]
mono_degree = [2, 2, 2, 2]
-> c_monomials = [1; x1; x2; y1; y2; x1^2; x1*x2; x1*y1; x1*y2; x2^2; x2*y1; x2*y2; y1^2; y1*y2; y2^2]
%}

% Define the coefficients of the array of helper polynomials
% for the CONVEX constraint
c_coef_help = sdpvar(k, length(c_monomials));
%{e.g., k=3, length(c_monomials)=15
c_coef_help = [c11, c12, c13, ..., c115;
               c21, c22, c23, ..., c215;
               c31, c32, c33, ..., c315]
%}

% Create an array of helper polynomials 
c_Q_help = c_coef_help*c_monomials;
%{c_Q_help = [
  c11*1 + c12*x1 + c13*x2 + c14*y1 + c15*y2 + c16*x1^2 + c17*x1*x2 + c18*x1*y1 + c19*x1*y2 + c110*x2^2 + c111*x2*y1 + c112*x2*y2 + c113*y1^2 + c114*y1*y2 + c115*y2^2;
  c21*1 + c22*x1 + c23*x2 + c24*y1 + c25*y2 + c26*x1^2 + c27*x1*x2 + c28*x1*y1 + c29*x1*y2 + c210*x2^2 + c211*x2*y1 + c212*x2*y2 + c213*y1^2 + c214*y1*y2 + c215*y2^2;
  c31*1 + c32*x1 + c33*x2 + c34*y1 + c35*y2 + c36*x1^2 + c37*x1*x2 + c38*x1*y1 + c39*x1*y2 + c310*x2^2 + c311*x2*y1 + c312*x2*y2 + c313*y1^2 + c314*y1*y2 + c315*y2^2;]
%}

%% PROBLEM SETUP: Write the constraints
t0 = tic(); % <- Start a timer
F = [sos(m_Q_help), sos(c_Q_help)]; % <- A=[2,2;3,3], B=[4;4] -> [A,B]= [2,2,4;3,3,4]
%{polynomial p(x) = x^4 + 2*x^2 + 1, sos(p) creates a constraint that ensures that p(x) is 'sum-of-squares'
%}

% Add monotonicity constraints
F = F+[sos(transpose(jacobian(p,x)).*monotone_profile - m_Q_help*transpose((x-inf_domain).*(sup_domain-x)))];
%{-> jacobian calculates the symbolic Jacobian df/dx of a polynomial f(x)
            e.g., x = sdpvar(1, 2), p = x(1)^2 + 3*x(1)*x(2) + x(2)^2 -> jacobian(p,x) = [2*x(1) + 3*x(2), 3*x(1) + 2*x(2)]
-> transpose: Transpose the Jacobi matrix into a k x 1 column vector -> transpose(jacobian(p,x))= [2*x(1) + 3*x(2); 3*x(1) + 2*x(2)]
-> .*: [2,2,4;3,3,4].*2=[4,4,8;6,6,8]
%}
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
%{The sum-of-squares decomposition gives p(x)=uT(x)Qu(x)=vT(x)v(x)
1. [sol,u,Q] = solvesos(Constraints,Objective,options,decisionvariables), where Constraints is F=sos(p) -> obtain u and Q, u'*Q*u is the sos of p
2. optimize(F); v = sosd(F), where F=sos(p) -> obtain v, and v'*v is the sos of p
%}
optimization_time = toc(t1);
msg = "Optimization runtime: " + optimization_time + " seconds.";
disp(msg);


%% Display message
msg = "Monotone-convex regression for polynomial of degree "+degree+" complete.";
aux_out = struct('setup_time', setup_time, 'optimization_time',...
    optimization_time, 'solver_time', sol.('solvertime'), 'train_rmse', sqrt(value(h)^2/N));
end

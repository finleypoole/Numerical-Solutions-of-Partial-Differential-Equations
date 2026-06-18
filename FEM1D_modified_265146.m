%{
***************************************************************************
845G1 - FEM 1D code 
Author: Chandrasekhar Venkataraman
adapted from code of
Cuneyt Sert
Date: 30.03.2013
***************************************************************************

Solves the following model PDE using linear finite elements in 1D.
-d/dx[a(x)*du/dx] + b(x)*du/dx + c(x)*u = f(x)
with two boundary conditions.


It is assumed that global nodes are ordered starting with 1 at the
minimum x value and increasing one by one upto NN = NE+1 corresponding
to the maximum x value.

***************************************************************************
Variables are:
***************************************************************************
USER INPUTS :

NE         : Number of elements.
NGP        : Number of GQ points used in numerical integration.
aFunc      : a(x) of the model DE.
bFunc      : b(x) of the model DE.
cFunc      : c(x) of the model DE.
fFunc      : f(x) of the model DE.
xMin, xMax : Minimum and maximum coordinates of the problem domain.
BCtype     : BC type of left (min x) and right (max x) boundaries.
             Array of size 2.
             1: EBC (Dirichlet), 2: NBC (Neumann)
EBCdata    : Array of 2. Stores PV for given EBCs, if there are any.
NBCdata    : Array of 2. Stores SV for given NBCs, if there are any.

OTHER VARIABLES :

NN         : Number of nodes of the mesh. It is NE+1 for linear elements.
coord      : Coordinates of mesh nodes, array of length NN.
GQpoint    : Gauss quadrature points, array of size NGP.
GQweight   : Gauss quadrature weights, array of size NGP.
ksi        : Coordinate on the reference element.
S          : Linear shape functions evaluated at GQ points. Array of size
             NGP.
dS         : Derivatives of shape functions evaluated at GQ points. Array
             of size NGP.
Ke         : Element level stiffness matrix of size 2x2.
Fe         : Element level force vector of size 2x1.
K          : Global stiffness matrix of size NNxNN. Although it is a sparse
             matrix, it is stored in full form for simplicity.
F          : Global force vector of size NN.
u          : Array of unknowns. Length NN.

%}




clc;
clear all;
close all;

% ************************************************************************
% Problem dependent data
% ************************************************************************

NE     =  50; %change this to see different mesh sizes     
NGP    =  3;  % number of quadrature points
epsilon    =  0.02;  %%added variable epsilon (FP) and new function
aFunc  =  'epsilon^2';   % Note that the 1st term of the solved DE has a minus sign. 
bFunc  =  '-1';
cFunc  =  '0';
fFunc  =  '1';
xMin   =  0;
xMax   =  1;

BCtype(1) = 1;  % 1: EBC, 2: NBC 
BCtype(2) = 1;  %%changed boundary conditions to dirichlet

EBCdata(1) = 0;
EBCdata(2) = 0;

NBCdata(1) = 0;
NBCdata(2) = 1;


% ************************************************************************
% Calculate node coordinates of the 1D mesh
% ************************************************************************

% Calculate the number of nodes (note linear elements)
NN = NE + 1;

range = xMax - xMin;

% Calculate nodal coordinates assuming uniform grid
he = range / NE;   % Length of each element.
for i = 1:NN
   coord(i) = xMin + (i-1) * he;
end




% ************************************************************************
% Enter Gauss Quadrature points and weights for different NGP values
% ************************************************************************
GQpoint  = zeros(NGP,1);
GQweight = zeros(NGP,1);

if NGP == 1  % One-point quadrature
   GQpoint(1) = 0;
   GQweight(1) = 2;
elseif NGP == 2  % Two-point quadrature
   GQpoint(1) = -sqrt(1/3);
   GQpoint(2) = sqrt(1/3);
   GQweight(1) = 1;
   GQweight(2) = 1;
elseif NGP == 3  % Three-point quadrature
   GQpoint(1) = -sqrt(3/5);
   GQpoint(2) = 0;
   GQpoint(3) = sqrt(3/5);
   GQweight(1) = 5/9;
   GQweight(2) = 8/9;
   GQweight(3) = 5/9;
end




% ************************************************************************
% Calculate shape functions and their derivatives at GQ points
% ************************************************************************

S  = zeros(2, NGP);
dS = zeros(2, NGP);

for i = 1:NGP
   S(1,i)  = 0.5 * (1 - GQpoint(i));
   S(2,i)  = 0.5 * (1 + GQpoint(i));
   dS(1,i) = -0.5; %Linear shape fnc.!
   dS(2,i) = 0.5;
end




% ************************************************************************
% Calculate global system matrix by the assembly of element matrices
% Note we transform to the reference elemement [-1,1] for integration
% ************************************************************************

K = zeros(NN, NN);
F = zeros(NN, 1);

for e = 1:NE
   % Intitialize Ke and Fe to zero.
   Ke = zeros(2,2);
   Fe = zeros(2,1);

   for k = 1:NGP   % Loop over quadrature points
      % Calculate global x coordinate that corresponds to local ksi.
      ksi = GQpoint(k);   % k-th GQ point
      % Calculate the x value inside element e that corresponds to ksi.
      x1e = coord(e);
      x2e = coord(e+1);
      x = 0.5 * (x2e - x1e) * ksi + 0.5 * (x1e + x2e);

      % Evaluate coefficients at this x value.
      aValue = eval(aFunc);
      bValue = eval(bFunc);
      cValue = eval(cFunc);
      fValue = eval(fFunc);

      % Calculate Jacobian of the element
      Jacob = 0.5 * (x2e - x1e);
      
      for i = 1:2
        for j = 1:2
           Ke(i,j) = Ke(i,j) + (aValue * dS(i,k)/Jacob * dS(j,k)/Jacob + ...
                                bValue * S(i,k) * dS(j,k)/Jacob + ...
                                cValue * S(i,k) * S(j,k)) * Jacob * GQweight(k);
        end
      end
      
      for i = 1:2
         Fe(i) = Fe(i) + S(i,k) * fValue * Jacob * GQweight(k);
      end
      
   end  % End of loop over all GQP
   
   % Assemble [Ke] into [K]
   K(e,e)     = K(e,e)     + Ke(1,1);
   K(e,e+1)   = K(e,e+1)   + Ke(1,2);
   K(e+1,e)   = K(e+1,e)   + Ke(2,1);
   K(e+1,e+1) = K(e+1,e+1) + Ke(2,2);

   % Assemble {Fe} into {F}
   F(e)   = F(e)   + Fe(1);
   F(e+1) = F(e+1) + Fe(2);

end  % End of element loop




% ************************************************************************
% Apply BCs
% ************************************************************************
% For EBCs [K] and {F} are modified
% SV values specified for NBCs are added to {F}.

% Left boundary
node = 1;

switch BCtype(1)
  case{1}   % If this BC is of EBC type
   F(node) = EBCdata(1);
   K(node,:) = 0.0;     % Equate node-th of [K] to zero.
   K(node,node) = 1.0;  % Equate diagonal of the node-th row of [K] to one.
   
  case{2}   % If this BC is of NBC type
   F(node) = F(node) + NBCdata(1);   % Add given SV value to {F}.
end


% Right boundary
node = NN;

switch BCtype(2)
  case{1}   % If this BC is of EBC type
   F(node) = EBCdata(2);
   K(node,:) = 0.0;     % Equate node-th row of [K] to zero.
   K(node,node) = 1.0;  % Equate diagonal of the node-th row of [K] to one.
   
  case{2}   % If this BC is of NBC type
   F(node) = F(node) + NBCdata(2);   % Add given SV value to {F}.
end




% ************************************************************************
% Solve the global system of [K]{u}={F}
% ************************************************************************

u = K\F;




% ************************************************************************
% Output the results and plot the solution
% ************************************************************************


% Write the calculated unknowns on the screen
disp('Calculated unknowns are:');
disp(u);

% Plot the nodal values as circles.
plot(coord, u, 'o', 'MarkerFaceColor', 'b');
hold(gca, 'on');

% Plot the solution over each element as lines. Straight lines are enough
% for linear elements.
for e = 1:NE
   plot (coord, u);  % Plot the linear solution over element e.
end

grid on;

 %Plot the exact solution if known
 x = xMin:(xMax-xMin)/500:xMax;
 %exact solution
 uexact = -x + (1 - exp(-x/epsilon^2)) / (1 - exp(-1/epsilon^2));
 plot(x, uexact, 'r'); %%changed the exact solution (FP)
 
%exact solution and errors
uexact_nodes = interp1(x, uexact, coord)';
abs_error = abs(u - uexact_nodes);

figure;
plot(coord, abs_error, 'b-o', 'MarkerFaceColor', 'b');
xlabel('x');
ylabel('|u - u_exact|');
title(['Pointwise Error, \epsilon = ', num2str(epsilon), ', NE = ', num2str(NE)]);
grid on;

%{
clc; clear all; close all;  
Plot for different values of epsilon
 x = 0:0.001:1
 eps_values = [1, 0.5, 0.2, 0.1, 1/50];

 hold on;
 for eps = eps_values
     u_exact = -x + (1 - exp(-x/eps^2)) / (1 - exp(-1/eps^2));
     plot(x, u_exact, 'LineWidth', 2, 'DisplayName', ['\epsilon = ', num2str(eps)]);
 end
 legend show;
 grid on;
%}

fprintf(1, '\nThe code  terminated properly.\n\n');


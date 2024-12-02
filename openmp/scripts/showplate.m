% 
% This Matlab script will slurp in the values for the temperatures on a plate
% discretized onto a (N+2) x (N+2) mesh. The outermost edges are all assumed
% to be boundary conditions, and this script can either plot just the  boundary
% conditions, or the full plate. If the full plate is shown, the boundary
% conditions are also displayed but using different colors.
%
% If you want to just see the boundary conditions, set
%       showBConly = true;
%       showall    = false; 
% If you want the full plate,
%       showBConly = false;
%       showall    = true; 
% Setting both to false will plot ... nothing, what did you expect?
%
% The input file should have two integers and a double on each row. The integers
% give the (i,j) coordinates of the corresponding value.
%
% Developed for B673,  2 Feb 2017.
%
%
T = load('plate');
i    = T(:,1);
j    = T(:,2);
vals = T(:,3);
N = max(i)-1;

showBConly = false;
showall    = true; 

% Can plot the whole damned thing :
if showall 
    positionfig
    plot3(i, j, vals, 'b+')
    xlabel('x')
    ylabel('y')
    zlabel('Temperature')
    title('Plate Temperature Values')
end

if (showBConly || showall)
    % First check the boundary conditions alone
    If = find(i == 0);
    Il = find(i == max(i));

    Jf = find(j == 0);
    Jl = find(j == max(j));

    if showBConly
        positionfig
        xlabel('Bottom')
        ylabel('Left-hand size')
        zlabel('Temperature')
        title(['Boundary nodes of plate: N = ' num2str(N)])
    else
        hold on
    end
    plot3(i(If), j(If), vals(If), 'c+', ...
          i(Il), j(Il), vals(Il), 'r+', ...
          i(Jf), j(Jf), vals(Jf), 'g*', ...
          i(Jl), j(Jl), vals(Jl), 'mo')
    hold off

end


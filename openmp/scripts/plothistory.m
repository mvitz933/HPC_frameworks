
G = load('history');
i = G(:,1);  % iteration numbers
e = G(:,2);  % error norm

positionfig
semilogy(i, e, 'b+')
xlabel('Iteration Number')
ylabel('Error norm')
title('Convergence History')


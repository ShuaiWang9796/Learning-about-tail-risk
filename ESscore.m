function [AL, FZG, NZ, AS] = ESscore(y, Q, ES, tau)
AL = mean(-log((tau-1)./ES) - (y-Q).*(tau-(y<=Q)) ./ (tau.*ES) + y./ES);
FZG = mean(((y<=Q)-tau) .* (Q-y+(Q.*exp(ES))./(tau.*(1+exp(ES)))) - exp(ES).*((y<=Q).*y./tau-ES)./(1+exp(ES)) - log(1+exp(ES)) + log(2));
%FZG = mean(((y<=Q)-tau).* Q - (y<=Q).*y + (ES./(1+exp(ES))) .* (ES-Q+(y<=Q).*(Q-y)./tau) + log(2./(1+exp(ES))));
NZ = mean(((y<Q)-tau).*Q./(2*tau.*(-ES).^0.5) - 1./(2.*(-ES).^0.5) .* ((y<Q).*y./tau - ES) +  (-ES).^0.5 );
AS = mean(tau * (ES.^2/2 + 2*Q.^2 - Q.*ES) + (y<=Q) .* (-ES .* (y-Q) + 2.*(y.^2 - Q.^2)));
end
function [AL] = ALscore(y, Q, ES, tau)
AL = sum(-log((tau-1)./ES) - (y-Q).*(tau-(y<=Q)) ./ (tau.*ES) + y./ES);
end
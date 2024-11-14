function QL = quantileloss(tau, Y, T)
   QL = mean(max(tau .* (T-Y),(1-tau).*(Y-T)));
end
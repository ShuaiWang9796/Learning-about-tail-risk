function val = QL(act, pre, tau)

val = mean((tau - (act<=pre)) .* (act-pre));
%max(tau*(act-pre), (tau-1)*(act-pre));

end


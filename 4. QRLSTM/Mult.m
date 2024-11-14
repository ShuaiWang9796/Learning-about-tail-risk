function z = Mult(x, Q)
for i = 1:length(Q)
    z(i,:) = (1 + exp(x))*Q(i);
end
end
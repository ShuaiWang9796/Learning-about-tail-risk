function z = Add(x, Q, y)
  z = 0;
for i = 2:length(Q)
 if y(i-1) <= Q(i-1)
    z(i,:) = x(1) + x(2)*(Q(i-1)-y(i-1)) + x(3)*z(i-1);
 else
    z(i,:) = z(i-1);
 end
end
end
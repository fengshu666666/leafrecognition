function feature = l2norm(feature)
den = sum(feature.^2,1);
den(den == 0) = 1;
feature = feature ./repmat(sqrt(den),size(feature,1),1);
end
function kval = rbf_new_kernel(u,v,rbf_sigma,varargin)


if nargin < 3 || isempty(rbf_sigma)
    rbf_sigma = 1;
else
    if ~isscalar(rbf_sigma) || ~isnumeric(rbf_sigma)
        error(message('stats:rbf_kernel:RBFSigmaNotScalar'));
    end
    if rbf_sigma == 0
        error(message('stats:rbf_kernel:SigmaZero'));
    end
    
end

dissq = repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))...
    -2*(u*v')+repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1);
dissq(dissq<0) = 0;


avesig = mean(mean(sqrt(dissq)));
rbf_sigma = rbf_sigma * avesig;
kval = exp(-(1/(2*rbf_sigma^2))*dissq);
end
function [fea,nfeature] = SPDFeature(data,opt)
%%% code from 
%%% Lei Wang, Jianjia Zhang, Luping Zhou, Chang Tang and Wanqing Li. 
%%% Beyond covariance: Feature representation with nonlinear kernel matrices. 
%%% In ICCV 2015

if(opt.datadim ~= size(data,2))
    data = data';
end
if(isfield(opt,'ratio'))
    data = data(1:opt.ratio:end,:);
end

nfeature = size(data,1);
switch upper(opt.SPDtype)
    case 'RBF'
        if(isfield(opt,'norm')&&opt.norm == 1)
            [~,dim] = size(data);
            data = data - repmat(mean(data,2),1,dim);
            data = l2norm(data);
        end
        fea = rbf_new_kernel(data',data',opt.rbf_sigma);
end







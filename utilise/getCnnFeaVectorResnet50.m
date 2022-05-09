function code = getCnnFeaVectorResnet50(datasetname,inputdatabase,net,conv_index)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTS:

%%% datasetname: Swedish (for example)

%%% inputdatabase: (for example training subset)
%%%     imnum: 375
%%%     cname: {1¡Á15 cell}
%%%     label: [375¡Á1 double]
%%%      path: {1¡Á375 cell}
%%%    nclass: 15

%%% net:
%%%                  layers: [1¡Á175 struct]
%%%                    vars: [1¡Á176 struct]
%%%                  params: [1¡Á215 struct]
%%%                   meta: [1¡Á1 struct]
%%%                    mode: 'test'
%%%                  holdOn: 0
%%%     accumulateParamDers: 0
%%%          conserveMemory: 0
%%%         parameterServer: []
%%%                  device: 'cpu'

%%% conv_index: 30 (for example)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%%% pca1x1 dimensionality reduction should be learned from 
%%% the training set in advance
switch datasetname
    case 'Swedish'
        load('pca1x1_Swedish_layer30_resnet50_V95.mat');
end

code = zeros(size(V,2)*(size(V,2)+1)/2,inputdatabase.imnum);

for ii = 1:inputdatabase.imnum    
    %%% read image
    I_ori = imread(inputdatabase.path{ii});
    I_ori = single(I_ori) ; %%% 255 range

    %%% preprocess image
    averageColour_ = mean(mean(net.meta.normalization.averageImage,1),2) ;   
    imageSize_ = net.meta.normalization.imageSize;        
    I = preprocessImage(I_ori, 1, imageSize_, averageColour_, 1, [0, 0]); 

    %%% cnn feature map
    net.mode = 'test' ;
    net.conserveMemory = 0;
    net.eval({'data', I}) ;        
    if length(conv_index)==1
        if conv_index==27
            feamap = net.vars(net.getVarIndex('res3a')).value ;
        end
        if conv_index==28
            feamap = net.vars(net.getVarIndex('res3b')).value ;
        end
        if conv_index==29
            feamap = net.vars(net.getVarIndex('res3c')).value ;
        end
        if conv_index==30
            feamap = net.vars(net.getVarIndex('res3d')).value ;
        end
        if conv_index==31
            feamap = net.vars(net.getVarIndex('res4a')).value ;
        end
        if conv_index==32
            feamap = net.vars(net.getVarIndex('res4b')).value ;
        end
        if conv_index==33
            feamap = net.vars(net.getVarIndex('res4c')).value ;
        end
        if conv_index==34
            feamap = net.vars(net.getVarIndex('res4d')).value ;
        end
        if conv_index==35
            feamap = net.vars(net.getVarIndex('res4e')).value ;
        end
        if conv_index==36
            feamap = net.vars(net.getVarIndex('res4f')).value ;
        end

        %%% pca 1x1 
        [h,w,c]=size(feamap);
        II = reshape(feamap,[h*w,c])';
        feamap = double(V'*II);
    end      
    sumvalue = sqrt(sum(feamap.^2,2));
    feamapNorm = feamap./repmat(sumvalue,1,size(feamap,2));
    feamapNorm(isnan(feamapNorm)) = 0;

    %%% kernel pooling     
    opt.norm = 1;
    opt.SPDtype = 'RBF';
    opt.datadim = size(feamapNorm,1);
    opt.rbf_sigma = 2.6;                
    SPDresult = SPDFeature(feamapNorm,opt);    

    %%% triu elements
    SPDresult = SPDresult+0.005*eye(size(SPDresult));        
    triuindex = triu(true(size(SPDresult,1)));
    SPDresult = sqrtm(SPDresult);
    features  = SPDresult(triuindex);
    code(:,ii) = features(:);
end







clc;clear;
%%
addpath(genpath(fullfile('matconvnet-1.0-beta25','matlab','mex')));
addpath(genpath('utilise'));
addpath(genpath('models'));

run(fullfile('matconvnet-1.0-beta25','matlab','vl_setupnn'));
%%% run(fullfile('matconvnet-1.0-beta25','matlab','vl_compilenn'));

%% database information
datasetname = 'Swedish'; 
imgformat = '*.tif';
database = retrDatabaseDir(fullfile('.','image',datasetname),imgformat);

fprintf('\n============== Experiments on %s ==============\n',datasetname);

net = dagnn.DagNN.loadobj('imagenet-resnet-50-dag');
conv_index = 30;

%%  training feature extraction
fprintf('\n------------Extract Train Feature via pretrained CNN ------------\n');
load('Swedish_pre_random_matrix.mat','pre_rand_matrix_tr');
tr_database = getSubBase(database,pre_rand_matrix_tr(1,:)); %%% pre_rand_matrix_tr contains 20 random splits

train_cov = getCnnFeaVectorResnet50(datasetname,tr_database,net,conv_index);
fprintf('\n------------Feature Length: %d, Train Num: %d----------\n',size(train_cov));
train_label = (tr_database.label)';

%% testing feature extraction and classification
fprintf('\n------------Extract Test Feature via pretrained CNN ------------\n');
load('Swedish_pre_random_matrix.mat','pre_rand_matrix_ts');%%% pre_rand_matrix_ts contains 20 random splits
ts_database = getSubBase(database,pre_rand_matrix_ts(1,:));
test_label = (ts_database.label)';

nCorrRecog = zeros(ts_database.imnum,2);
predict_label = zeros(ts_database.imnum,2);

for idx = 1:ts_database.imnum   
    one_ts_database.imnum=1;
    one_ts_database.path=ts_database.path(idx);
    one_test_label = test_label(idx);

    test_cov = getCnnFeaVectorResnet50(datasetname,one_ts_database,net,conv_index);

    [predict_label(idx,1),out_NN]  = NNClassifier(train_cov,test_cov,train_label,one_test_label,'euclidean');
    [predict_label(idx,2),out_NNC] = NNClassifier(train_cov,test_cov,train_label,one_test_label,'cosine');

    if out_NN==1
        nCorrRecog(idx,1) = nCorrRecog(idx,1)+1;
    end
    if out_NNC==1
        nCorrRecog(idx,2) = nCorrRecog(idx,2)+1;
    end

    if 0==mod(idx,ts_database.imnum/10)||idx==ts_database.imnum
        fprintf('Accuracy up to %05d tests is nn(%.2f%%), nncos(%.2f%%)\n',[idx 100*sum(nCorrRecog,1)/idx]); 
    end
end



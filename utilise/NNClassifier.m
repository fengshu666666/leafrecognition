function [ predict_label,all_rate ] = NNClassifier(train_matr,test_matr,label_Train,label_Test,metric)

%%% train_matr   - D*N each colomn is a sample
%%% test_matr    - D*N each colomn is a sample
%%% label_Train  - 1*N
%%% label_Test   - 1*N

dist = pdist2(test_matr',train_matr',metric);
%%% ====== metric: ======
%%% 'euclidean' / 'sqeuclidean' / 'chisq' / 'cosine' / 'emd'/ 'L1'

predict_label = nn(dist, label_Train);
all_rate = sum(predict_label(:)==label_Test(:))/length(label_Test);



function predict_label = nn(dist, tr_label)

[~, indexs] = sort(dist, 2, 'ascend');

predict_label = zeros(1, size(indexs, 1));
for i=1:size(indexs, 1)
    predict_label(i) = tr_label(indexs(i, 1));
end












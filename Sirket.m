clc
clear all
close all
df = readtable("data.csv");
y=df.Bankrupt_;

x=df(:,2:96);



x = normalize(x);
x = removevars(x, 'Liability_AssetsFlag');
x = removevars(x, 'NetIncomeFlag');



%verileri pythondaki random state fonksiyonu işlevini gören algoritma ile
%rastgele parçalama işlemi(öğrenme kısmında birden fazla kodu calıştırma işlemi yapılmalı verilerin 1 classta toplanma ihtimaline karşı) 
rng(42); 
indices = randperm(size(x, 1));
split_point = round(0.3 * size(x, 1));
x_train = x(indices(1:split_point), :);
x_test = x(indices(split_point+1:end), :);
y_train = y(indices(1:split_point));
y_test = y(indices(split_point+1:end));

m1 = min(x_train{:,:});
m2 = max(x_train{:,:});
range = [m1' m2']; % Transpozunu almayı unutmayın
net = newff(range,[93 22 1],{'tansig','tansig','logsig'},'trainlm');
net.trainparam.show = 25;
net.trainparam.epochs = 50;
net.trainparam.goal = 1e-12;
net.trainparam.maxfail = 60;
net.trainparam.memreduc = 1;
net = train(net,x_train{:,:}',y_train'); % Transpozunu almayı unutmayın
y = net(x_train{:,:}'); % Transpozunu almayı unutmayın
perf = perform(net,y,y_train);


plotconfusion(y_train',y)
title("train Data")
 
% test data
res_test = net(x_test'); 
perf = perform(net,res_test,y_test)
 
figure
 plotconfusion(y_test',res_test)
title("test Data")








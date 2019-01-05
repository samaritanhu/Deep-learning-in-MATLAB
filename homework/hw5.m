% Homework 5 for Deeplearning 2018 Fall
%
% You need to:
%   * Implement the backwardLoss function
%
% Submission:
%   * run this file in the server; everything else is done automatically
% 
% Dec 21, 2018 by
% Johnny Chen <johnnychen94@hotmail.com>

%%%%% BEGIN of demo %%%%%

[X,Y] = readData_iris(); % Y is one-hot encoding
width = 20; % number of neurons of hidden layer
[W1, W2] = initializeNetwork(size(X,1), width, size(Y,1));
trainingOptions = struct(...
    'InitialLearnRate', 1e-3, ...
    'LearnRateDropPeriod', 7000, ...
    'MaxEpoch',10000, ...
    'LearnRateDropFactor', 0.1, ...
    'Verbose', true, ...
    'VerboseFrequency', 100);
[W1, W2] = trainNetwork(X, Y, W1, W2, trainingOptions);
Y_hat = predict(X,W1, W2);

% show results
n = size(X,2);
figure;
plot(1:n, onehot2gray(Y),'ro',...
    1:n, onehot2gray(Y_hat),'b+');
xlabel('x');ylabel('y');title('prediction');

fprintf(sprintf("Accuracy: %.2f\n",100*accuracy(Y_hat, Y)));

% save results
submit_homework(W1,W2);
%%%%% END of demo %%%%%

%%%% BEGIN of Homework %%%%
% backwardLoss is the oooooooonly function you need to modify/implement
function [dLdW1, dLdW2] = backwardLoss(X,V1,Y1,V2,Y2,Y,W1,W2)
% backwardloss calculates the gradient
    m = size(Y, 2);
    dphidx = @(x) exp(x)./((1 + exp(x)).*(1 + exp(x)));
    dV2 = (Y2 - Y);
    dLdW2 = (dV2 * Y1')';
    dY1 = W2';
    dY2 = (Y2 - Y)./(Y2 .* (1 - Y2));
    dV1 = dphidx(V1);
    dW1 = X';
    dLdW1 = (dY1'* dV2 .* dV1 * dW1)';
    %dLdW1 = (dY1'* dY2 .* dV1 * dW1 / m)';
%error("Implement this to finish HW5");
end

%%%% END of Homework %%%%

function [W1, W2] = trainNetwork(X, Y, W1, W2, option)
   InitialLearnRate    = option.InitialLearnRate;
   LearnRateDropPeriod = option.LearnRateDropPeriod;
   LearnRateDropFactor = option.LearnRateDropFactor;
   MaxEpoch            = option.MaxEpoch;
   Verbose             = option.Verbose;
   VerboseFrequency    = option.VerboseFrequency;
   Momentum            = 0.9;
   
   if Verbose     
       hfig = figure;

       subplot(1,2,1);
       hfig_loss = animatedline();
       xlabel('iter');ylabel('Mean Squared Error');
       title('Gradient Descent Progress')
   end
    
   LearnRates = LearnRateScheduler(InitialLearnRate, MaxEpoch, LearnRateDropPeriod, LearnRateDropFactor);
   NumSample = size(X,2);
   v1 = zeros(size(W1)); v2 = zeros(size(W2));
   for epoch = 1:MaxEpoch
       % get training data of current iteration
       lr = LearnRates(epoch);
       
       % training
       [V1, Y1, V2, Y2] = forward(X, W1, W2);
       loss = crossentropy(Y2, Y);
       [dLdW1, dLdW2] = backwardLoss(X,V1,Y1,V2,Y2,Y,W1,W2);
       [W1, W2, v1, v2] = update_sgdm(W1, W2, dLdW1, dLdW2, v1, v2, lr, Momentum);
       
       % show training progress
       if Verbose && mod(epoch, VerboseFrequency) == 0
           addpoints(hfig_loss, epoch, loss);
           
           figure(hfig);subplot(1,2,2);
           plot(1:NumSample,onehot2gray(Y),'ro',...
               1:NumSample,onehot2gray(Y2),'b+');
           xlabel('x');ylabel('y');title('prediction');
           
           drawnow;
           
           fprintf(sprintf("Epoch: %d\tLoss: %f\tLearning Rate: %f\n", epoch, loss, lr));
       end
   end
end


function [W1,W2] = initializeNetwork(inLength,width,outLength)
% initNet returns initial weights for the so-called "network"
    W1 = 1e-3 .* rand(inLength,width);
    W2 = 1e-3 .* rand(width,outLength);
end

function [V1, Y1, V2, Y2] = forward(X, W1, W2)
% forward outputs the network result with given input X
    V1 = W1' * X;
    Y1 = logistic(V1);
    V2 = W2' * Y1;
    Y2 = softmax(V2);
end

function Y = predict(X, W1, W2)
% predict predicts the label of X with given network weight W 
    [~,~,~,Y] = forward(X, W1, W2);
    Y = round(Y);
end

function rst = accuracy(Y, Y_hat)
    Y = onehot2gray(round(Y));
    Y_hat = onehot2gray(round(Y_hat));
    rst = 1 - sum(Y_hat ~= Y)/numel(Y);
end

function loss = crossentropy(Y, T)
% Y: predict result
% T: label
    numElems = size(Y,2);
    loss = -sum( sum(T.* log(Y)) ./numElems );
end

function Y = softmax(X)
    X = X - max(X,[],1);
    expX = exp(X);
    Y = expX./sum(expX);
end

function y = logistic(x)
    y = 1 ./ (1 + exp(-x));
end

function [W1, W2, v1, v2] = update_sgdm(W1, W2, dLdW1, dLdW2, v1, v2, lr, mom)
% update_sgdm updates network weights W using gradient descent with momentum
%
% Note: the nesterov momentum version is used
    v_prev = v1;
    v1 = mom * v1 - lr * dLdW1;
    W1 = W1 - mom * v_prev + (1+mom) * v1;
    
    v_prev = v2;
    v2 = mom * v2 - lr * dLdW2;
    W2 = W2 - mom * v_prev + (1+mom) * v2;
end

function gray = onehot2gray(onehot)
% onehot2gray converts onehot encoding to gray encoding
%
% This function is particularly useful for multi-class classification:
%   * convert onehot-encoding training labels to gray-encoding
%   * convert onehot-encoding network output to gray-encoding for easy visualization.
%
% Examples:
%   for exact onehot encoding:
%       [1,0,0] --> 1
%       [0,1,0] --> 2
%       [0,1,1] --> 3
%   for predicted onehot encoding:
%       [0.9, 0.1, 0.3] --> 0.9
%       [0.1, 0.9, 0.3] --> 1.9
%       [0.1, 0.3, 0.9] --> 2.9
%
% See also: gray2onehot
    onehot(onehot<0) = 0;
    onehot(onehot>1) = 1;
    
    [v,i] = max(onehot);
    gray  = v + i - 1;
end

%%%% You are not expected to understand the following codes

function lr = LearnRateScheduler(InitialLearnRate, MaxEpoch, LearnRateDropPeriod, LearnRateDropFactor)
% LearnRateScheduler generates a sequence of learning rate using step decay method
    lr = repmat(InitialLearnRate,MaxEpoch,1);
    n_drop = ceil(MaxEpoch/LearnRateDropPeriod);
    for i = 2:n_drop
        pos_l = (i-1)*LearnRateDropPeriod + 1;
        pos_r = min(i*LearnRateDropPeriod, MaxEpoch);
        lr(pos_l:pos_r) = lr(pos_l-1) * LearnRateDropFactor;
    end
end

function [X,Y] = readData_iris()
% readData_iris reads iris dataset
%
% See also: readData
    dataurl    = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
    datadir    = 'data';
    filename   = fullfile(datadir,'Iris.txt');
    
    if ~isfolder(datadir)
        mkdir(datadir);
    end
    if ~isfile(filename)
        websave(filename, dataurl);
    end

    assert(isfile(filename), [filename,' not found']);
    
    data = readtable(filename);
    X = table2array(data(:,1:4))';
    Y = table2cell(data(:,5)); % convert table to cell
    Y = cellfun(@labelclass, Y, 'UniformOutput', false); % convert string to vector
    Y = cell2mat(Y)'; % convert cell to matrix
    return 
    
    function num = labelclass(str) % onehot encoding
        switch str
            case 'Iris-setosa'
                num = [1,0,0];
            case 'Iris-versicolor'
                num = [0,1,0];
            case 'Iris-virginica'
                num = [0,0,1];
            otherwise
                error(['unrecognized iris class', str])
        end
    end
end

function submit_homework(W1, W2)
    rootpath = '~/Homework/hw5';
    mkdir(rootpath);
    if strcmp(computer('arch'),'glnxa64')
        % save weights
        file_W = fullfile(rootpath, 'W.mat');
        W = {W1, W2};
        save(file_W, 'W');

        % save codes
        file_code = fullfile(rootpath, 'hw5.m');
        call = sprintf("cp %s %s", [mfilename('fullpath'),'.m'], file_code);
        status = system(call);
        
        if status~=0
            error("Submit failed!")
        else
            fprintf("Submit success!\n");
        end
    else
        fprintf("To submit, you need to run this file in the server\n");
    end
end


%Brittani McKann

load samplecredit.mat

%Part I
% Use the first 500 data for the
% training set and the rest for the testing set.
% implement the Perceptron Learning Algorithm for
% this data set. Set the number of epochs to be 1000. For the output, report the accuracy for the
% training data as well as the testing data 

%training set from sample
Apps = [ones(500,1),sample(1:500,:)];
Dec = [label(1:500,:)];

%Define w
w = [rand(1,16)-.5]; %Generated until accuracy improved
%w = [-.3436,.3555,.1448,-.1237, -.3091,-.0717,-.0179,-.3794,.0895,-.2738,-.1154,.0829,-.2482,-.2096,.1171,.2347];
u = w'; %store original w to use in normalized data
w = w';

%number of epochs
E = 1000;

%calculate correct w
k = 1;
while k <= E  
    for i = 1:500 %runs through all applications
        if sum(w'.*Apps(i,:)) < 0 & Dec(i,:) > 0 %checks if sign doesn't match
            w = [w + Dec(i,:)*Apps(i,:)']; %w_new=w_guess+y(i)+x(i)
        elseif sum(w'.*Apps(i,:)) > 0 & Dec(i,:) < 0 %checks other case of sign mismatch
            w = [w + Dec(i,:)*Apps(i,:)']; %w_new=w_guess+y(i)*x(i)
        end
    end  
    k = k+1; %moves to next iteration
end

%calculate decisions on training data with weights
AppsDec = w'.*Apps; %AppsDec => Decisions on training data
AppsDec = AppsDec'; %Puts features in column position
AppsDec = [sum(AppsDec)]'; % w_1*x_1+...+w_d*x_d

%Simplifies values to +/- 1 in AppsDec(like using the sign function)
for i = 1:length(AppsDec)
    if AppsDec(i) < 0
        AppsDec(i) = -1;
    else
        AppsDec(i) = 1;
    end
end

%calculate accuracy of the weights on training data
Tcounter = 0; %training decision counter
for i = 1:length(AppsDec)
    if AppsDec(i) == Dec(i)
        Tcounter = Tcounter+1;
    else
        Tcounter = Tcounter+0;
    end
end
TrainAcc = Tcounter/length(AppsDec) %Accuracy of the decision on training data


%test on unused data
TestApps = [ones(153,1),sample(501:end,:)];

%Find decsion on test applications
TestDec = w'.*TestApps;
TestDec = TestDec'; %Puts features in column position
TestDec = [sum(TestDec)]'; % w_1*x_1+...+w_d*x_d

%Real decision on test applications
RealDec = [label(501:end,:)];

%Simplifies values to +/- 1 in TestDec(like using the sign function)
for i = 1:length(TestDec)
    if TestDec(i) < 0
        TestDec(i) = -1;
    else
        TestDec(i) = 1;
    end
end

%counts how many times decision was correct
counter = 0;
for i = 1:length(TestDec)
    if TestDec(i) == RealDec(i)
        counter = counter+1;
    else
        counter = counter+0;
    end
end

%calculates percent of the time correct
Accuracy = counter/length(TestDec)

%PartII
%Normalized training data Perceptron
%sample normalized
NApps = [];

for i = 1:15
    max_value = max(sample(:,i));
    NApps(:,i) = sample(:,i)/max_value;
end

%pulling training apps from the normalized data
TrainNormApps = [ones(500,1),NApps(1:500,:)];

%weights for normalized
Nw = u; 

%calculate correct w
k = 1;
while k <= E %reuses same E
    for i = 1:500 %runs through all applications
        if sum(Nw'.*TrainNormApps(i,:)) < 0 & Dec(i,:) > 0 %checks if sign doesn't match
            Nw = [Nw + Dec(i,:)*TrainNormApps(i,:)']; %w_new=w_guess+y(i)+x(i)
        elseif sum(Nw'.*TrainNormApps(i,:)) > 0 & Dec(i,:) < 0 %checks other case of sign mismatch
            Nw = [Nw + Dec(i,:)*TrainNormApps(i,:)']; %w_new=w_guess+y(i)+x(i)
        end
    end  
    k = k+1; %moves to next iteration
end

%Calculate decisions on normalized training data with weight vector
TrainNormAppsDec = Nw'.*TrainNormApps; % Apply weights to normalized training data
TrainNormAppsDec = TrainNormAppsDec'; % set columns to credit applications
TrainNormAppsDec = [sum(TrainNormAppsDec)]'; % sum the weights*features of each application  then turn it into a column vector

%Simplifies values to +/- 1 in normalized TrainNormAppsDec (sign function)
for i=1:length(TrainNormAppsDec)
    if TrainNormAppsDec(i)<0
        TrainNormAppsDec(i)=-1;
    else
        TrainNormAppsDec(i)=1;
    end
end

%calculate accuracy of the training weights on normalized training data
TNAcounter = 0; % Normalized application data correct decision counter
for i = 1:length(TrainNormAppsDec)
    if TrainNormAppsDec(i) == Dec(i)
        TNAcounter = TNAcounter+1;
    else
        TNAcounter = TNAcounter+0;
    end
end

%Calculate the accuracy of the decisions on normalized training data
NormTrainAccuracy = TNAcounter/length(TrainNormAppsDec)

%test on unnormalized unused data
NTestApps = [ones(153,1),sample(501:end,:)];

%Find decsion on test applications
NTestDec = Nw'.*NTestApps; %apply weights
NTestDec = NTestDec'; %oreints features into the columns
NTestDec = [sum(NTestDec)]'; %Sums features

%Simplifies values to +/- 1 in normalized TestDec (sign function
for i=1:length(NTestDec)
    if NTestDec(i)<0
        NTestDec(i)=-1;
    else
        NTestDec(i)=1;
    end
end

%counts how many times decision was correct
Ncounter = 0;
for i = 1:length(NTestDec)
    if NTestDec(i) == RealDec(i)
        Ncounter = Ncounter+1;
    else
        Ncounter = Ncounter+0;
    end
end
%calculates percent of the time correct with normalized data
NwAccuracy = Ncounter/length(NTestDec)

%test Nw on normalized testing data
NormTestApps = [ones(153,1), NApps(501:end,:)];

%Calculate decisions
NormTestDec = Nw'.*NormTestApps;
NormTestDec = NormTestDec';
NormTestDec = [sum(NormTestDec)]';

%replace values with +/- 1
for i=1:length(NormTestDec)
    if NormTestDec(i)<0
        NormTestDec(i)=-1;
    else
        NormTestDec(i)=1;
    end
end

%Calculate how many times correct
normcounter = 0;
for i = 1:length(NormTestDec)
    if NormTestDec(i) == RealDec(i)
        normcounter = normcounter+1;
    else
        normcounter = normcounter+0;
    end
end
%accuracy with norm weight vector and norm test data
BothNormAcc = normcounter/length(NormTestDec)

%Decisions using unchanged training data to calculate weights
NormTesting_UnchangedTrain = w'.*NormTestApps;
NormTesting_UnchangedTrain = NormTesting_UnchangedTrain';
NormTesting_UnchangedTrain = [sum(NormTesting_UnchangedTrain)]';

%Change +/- 1 entries
for i=1:length(NormTesting_UnchangedTrain)
    if NormTesting_UnchangedTrain(i)<0
        NormTesting_UnchangedTrain(i)=-1;
    else
        NormTesting_UnchangedTrain(i)=1;
    end
end

%Count how many times correct
NTestOnlyCount = 0;
for i = 1:length(NormTesting_UnchangedTrain)
    if NormTesting_UnchangedTrain(i) == RealDec(i)
        NTestOnlyCount = NTestOnlyCount+1;
    else
        NTestOnlyCount = NTestOnlyCount+0;
    end
end
%accuracy with unchanged training data but normalized test data
NormTestOnlyAcc = NTestOnlyCount/length(NormTesting_UnchangedTrain)
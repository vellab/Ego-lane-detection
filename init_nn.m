%% get the digit data
clear all

%load fruit_train
%load fruit_valid
load Encoded_umm;


%% initialize the net structure.
num_inputs = size(inputs_train, 1);
num_hiddens = 110;
num_outputs = 1;

%%% make random initial weights smaller, and include bias weights
W1 = 0.01 * randn(num_inputs, num_hiddens);
b1 = zeros(num_hiddens, 1);
W2 = 0.01 * randn(num_hiddens, num_outputs);
b2 = zeros(num_outputs, 1);

dW1 = zeros(size(W1));
dW2 = zeros(size(W2));
db1 = zeros(size(b1));
db2 = zeros(size(b2));

eps = 0.05;  %% the learning rate 
momentum = 0;   %% the momentum coefficient

num_epochs = 350; %% number of learning epochs (number of passes through the
                 %% training set) each time runbp is called.

total_epochs = 0; %% number of learning epochs so far. This is incremented 
                    %% by numEpochs each time runbp is called.

%%% For plotting learning curves:
min_epochs_per_plot = 200;
train_errors = zeros(1, min_epochs_per_plot);
valid_errors = zeros(1, min_epochs_per_plot);
epochs = [1 : min_epochs_per_plot];



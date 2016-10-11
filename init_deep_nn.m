%% get the digit data


%load fruit_train
%load fruit_valid
load Encoded_umm;


%% initialize the net structure.
num_inputs = size(inputs_train, 1);
num_hiddens1 = 35;
num_hiddens2 = 35;
num_hiddens3 = 35;


num_outputs = 1;

%%% make random initial weights smaller, and include bias weights
W1 = 0.01 * randn(num_inputs, num_hiddens1);
b1 = zeros(num_hiddens1, 1);
W2 = 0.01 * randn(num_hiddens1, num_hiddens2);
b2 = zeros(num_hiddens2, 1);
W3 = 0.01 * randn(num_hiddens2, num_hiddens3);
b3 = zeros(num_hiddens3, 1);
W4 = 0.01 * randn(num_hiddens3, num_outputs);
b4 = zeros(num_outputs, 1);

dW1 = zeros(size(W1));
dW2 = zeros(size(W2));
dW3 = zeros(size(W3));
dW4 = zeros(size(W4));
db1 = zeros(size(b1));
db2 = zeros(size(b2));
db3 = zeros(size(b3));
db4 = zeros(size(b4));


batch_size=1000;
eps = 0.1;  %% the learning rate 
momentum = 0.5;   %% the momentum coefficient

num_epochs = 350; %% number of learning epochs (number of passes through the
                 %% training set) each time runbp is called.

total_epochs = 0; %% number of learning epochs so far. This is incremented 
                    %% by numEpochs each time runbp is called.

%%% For plotting learning curves:
min_epochs_per_plot = 200;
train_errors = zeros(1, min_epochs_per_plot);
valid_errors = zeros(1, min_epochs_per_plot);
epochs = [1 : min_epochs_per_plot];



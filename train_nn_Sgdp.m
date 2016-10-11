%% To run this program:
%%   First run initnn.m
%%   Then repeatedly call train_nn.m until convergence.

train_CE_list = zeros(1, num_epochs);
valid_CE_list = zeros(1, num_epochs);
train_count_list = zeros(1, num_epochs);
valid_count_list = zeros(1, num_epochs);

start_epoch = total_epochs + 1;

%%%%%%%%%%%%%%%%%%%
eps=0.00001;
momentum=0;
%%%%%%%%%%%%%%%%%%%%

num_train_cases = size(inputs_train, 2);
num_valid_cases = size(inputs_valid, 2);

for epoch = 1:num_epochs
  % Selection of inputs that are to be sent into SGD
  tot_train=num_train_cases;
  %Permutated indices for all the training cases
  rp = randperm(tot_train);
  %Number of full batches possible in the training set
  multiple=floor(tot_train/batch_size);
  k=batch_size;
  for s=1:k:multiple*k
  inputs_for_iter=inputs_train(:,rp(s:s+k-1));
  targets_for_iter=target_train(rp(s:s+k-1));
  % Fprop
  h1_input = W1' * inputs_for_iter + repmat(b1, 1, k);  % Input to first hidden layer.
  h1_output = 1 ./ (1 + exp(-h1_input));  % Output of first hidden layer.
  h2_input = W2' * h1_output + repmat(b2, 1, k);  % Input to second hidden layer.
  h2_output = 1 ./ (1 + exp(-h2_input));  % Output of second hidden layer.
  h3_input = W3' * h2_output + repmat(b3, 1, k);  % Input to third hidden layer.
  h3_output = 1 ./ (1 + exp(-h3_input));  % Output of third hidden layer.
  logit = W4' * h3_output + repmat(b4, 1, k);  % Input to output layer.
  prediction = 1 ./ (1 + exp(-logit));  % Output prediction.

  % Compute cross entropy
  train_CE = -mean(mean(targets_for_iter .* log(prediction) + (1 - targets_for_iter) .* log(1 - prediction)));

  countwr_train=0;
  % Compute classification error for training inputs
  for i=1:size(targets_for_iter,2)
      if(targets_for_iter(i))
          if(prediction(i)<0.5)
              countwr_train=countwr_train+1;
          end
      else
          if(prediction(i)>0.5)
              countwr_train=countwr_train+1;
          end
      end
  end
  %Normalise count
  countwr_train=countwr_train/k;
  % Compute deriv
  dEbydlogit = prediction - targets_for_iter;

  % Backprop
  dEbydh3_output = W4 * dEbydlogit;
  dEbydh3_input = dEbydh3_output .* h3_output .* (1 - h3_output) ;
  dEbydh2_output = W3 * dEbydh3_input;
  dEbydh2_input = dEbydh2_output .* h2_output .* (1 - h2_output) ;
  dEbydh1_output = W2 * dEbydh2_input;
  dEbydh1_input = dEbydh1_output .* h1_output .* (1 - h1_output) ;

  % Gradients for weights and biases.
  dEbydW4 = h3_output * dEbydlogit';
  dEbydb4 = sum(dEbydlogit, 2);
  dEbydW3 = h2_output * dEbydh3_input';
  dEbydb3 = sum(dEbydh3_input, 2);
  dEbydW2 = h1_output * dEbydh2_input';
  dEbydb2 = sum(dEbydh2_input, 2);
  dEbydW1 = inputs_for_iter * dEbydh1_input';
  dEbydb1 = sum(dEbydh1_input, 2);

  %%%%% Update the weights at the end of the epoch %%%%%%
  dW1 = momentum * dW1 - (eps / k) * dEbydW1;
  dW2 = momentum * dW2 - (eps / k) * dEbydW2;
  dW3 = momentum * dW3 - (eps / k) * dEbydW3;
  dW4 = momentum * dW4 - (eps / k) * dEbydW4;
 
  db1 = momentum * db1 - (eps / k) * dEbydb1;
  db2 = momentum * db2 - (eps / k) * dEbydb2;
  db3 = momentum * db3 - (eps / k) * dEbydb3;
  db4 = momentum * db4 - (eps / k) * dEbydb4;
  
  W1 = W1 + dW1;
  W2 = W2 + dW2;
  W3 = W3 + dW3;
  W4 = W4 + dW4;
  
  
  b1 = b1 + db1;
  b2 = b2 + db2;
  b3 = b3 + db3;
  b4 = b4 + db4;
 
  

  %%%%% Test network's performance on the valid patterns %%%%%
  h1_input = W1' * inputs_valid + repmat(b1, 1, num_valid_cases);  % Input to hidden layer.
  h1_output = 1 ./ (1 + exp(-h1_input));  % Output of hidden layer.
  h2_input = W2' * h1_output + repmat(b2, 1, num_valid_cases);  % Input to hidden layer.
  h2_output = 1 ./ (1 + exp(-h2_input));  % Output of hidden layer.
  h3_input = W3' * h2_output + repmat(b3, 1, num_valid_cases);  % Input to third hidden layer.
  h3_output = 1 ./ (1 + exp(-h3_input));  % Output of third hidden layer.
  logit = W4' * h3_output + repmat(b4, 1, num_valid_cases);  % Input to output layer.
  prediction = 1 ./ (1 + exp(-logit));  % Output prediction.
  
  valid_CE = -mean(mean(target_valid .* log(prediction) + (1 - target_valid) .* log(1 - prediction)));

  countwr_valid=0;
  % Compute classification error for training inputs
  for i=1:size(target_valid,2)
      if(target_valid(i))
          if(prediction(i)<0.5)
              countwr_valid=countwr_valid+1;
          end
      else
          if(prediction(i)>0.5)
              countwr_valid=countwr_valid+1;
          end
      end
  end
  %Normalise count
  countwr_valid=countwr_valid/num_valid_cases;
  %%%%%% Print out summary statistics at the end of the epoch %%%%%
  
  train_CE_list(1, epoch) = train_CE;
  valid_CE_list(1, epoch) = valid_CE;
  train_count_list(1,epoch)=countwr_train;
  valid_count_list(1,epoch)=countwr_valid;
  fprintf(1,'%d  Train CE=%f, Valid CE=%f, Train Error=%f, Valid Error=%f\n',...
            total_epochs, train_CE, valid_CE, countwr_train, countwr_valid);
  end
%   %Cut the learning rate to a half after a epoch
%   eps=eps/2;
  total_epochs = total_epochs + 1;
  if total_epochs == 1
      start_error = train_CE;
  end
end

clf; 
if total_epochs > min_epochs_per_plot
  epochs = [1 : total_epochs];

end

% %%%%%%%%% Plot the learning curve for the training set patterns CE %%%%%%%%%
train_errors(1, start_epoch : total_epochs) = train_CE_list;
valid_errors(1, start_epoch : total_epochs) = valid_CE_list;
  subplot(2,1,1);
  hold on, ...
  plot(epochs(1, 1 : total_epochs), train_errors(1, 1 : total_epochs), 'b'),...
  plot(epochs(1, 1 : total_epochs), valid_errors(1, 1 : total_epochs), 'g'),...
  legend('Train', 'Test'),...
  title('Cross Entropy vs Epoch'), ...
  xlabel('Epoch'), ...
  ylabel('Cross Entropy');
  

%%%%%%%%% Plot the learning curve for the training set patterns count %%%%%%%%%
train_counts(1, start_epoch : total_epochs) = train_count_list;
valid_counts(1, start_epoch : total_epochs) = valid_count_list;
  subplot(2,1,2);
  hold on, ...
  plot(epochs(1, 1 : total_epochs), train_counts(1, 1 : total_epochs), 'b'),...
  plot(epochs(1, 1 : total_epochs), valid_counts(1, 1 : total_epochs), 'g'),...
  legend('Train', 'Test'),...
  title('Classification Error vs Epoch'), ...
  xlabel('Epoch'), ...
  ylabel('Classification Error');

%% To run this program:
%%   First run initnn.m
%%   Then repeatedly call train_nn.m until convergence.

train_CE_list = zeros(1, num_epochs);
valid_CE_list = zeros(1, num_epochs);
train_count_list = zeros(1, num_epochs);
valid_count_list = zeros(1, num_epochs);

start_epoch = total_epochs + 1;
%Testing if eps can be changed in the middle
eps=0.002;
momentum=0.1;
num_epochs = 3000;
%%%%%%%%%%%%%%%%%%%%

num_train_cases = size(inputs_train, 2);
num_valid_cases = size(inputs_valid, 2);

for epoch = 1:num_epochs
  % Fprop
  h_input = W1' * inputs_train + repmat(b1, 1, num_train_cases);  % Input to hidden layer.
  h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
  logit = W2' * h_output + repmat(b2, 1, num_train_cases);  % Input to output layer.
  prediction = 1 ./ (1 + exp(-logit));  % Output prediction.

  % Compute cross entropy
  train_CE = -mean(mean(target_train .* log(prediction) + (1 - target_train) .* log(1 - prediction)));

  countwr_train=0;
  % Compute classification error for training inputs
  for i=1:size(target_train,2)
      if(target_train(i))
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
  countwr_train=countwr_train/num_train_cases;
  %%%%%%%%%%%%%%%%%%%
  %For Precision
  countsaid=0;
  countfound=0;
  for i=1:size(target_train,2)
      if(prediction(i)>0.5)
          countsaid=countsaid+1;
          if(target_train(i))
              countfound=countfound+1;
          end
      end
  end
  precision_train=countfound/countsaid;
  %For Recall
  countthere=0;
  for i=1:size(target_train,2)
      if(target_train(i))
          countthere=countthere+1;
      end
  end  
  recall_train=countfound/countthere;
  f1_train=2*precision_train*recall_train/(precision_train+recall_train);
  %%%%%%%%%%%%%%%%%%%
  % Compute deriv
  dEbydlogit = prediction - target_train;

  % Backprop
  dEbydh_output = W2 * dEbydlogit;
  dEbydh_input = dEbydh_output .* h_output .* (1 - h_output) ;

  % Gradients for weights and biases.
  dEbydW2 = h_output * dEbydlogit';
  dEbydb2 = sum(dEbydlogit, 2);
  dEbydW1 = inputs_train * dEbydh_input';
  dEbydb1 = sum(dEbydh_input, 2);

  %%%%% Update the weights at the end of the epoch %%%%%%
  dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1;
  dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2;
  db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1;
  db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2;

  W1 = W1 + dW1;
  W2 = W2 + dW2;
  b1 = b1 + db1;
  b2 = b2 + db2;

  %%%%% Test network's performance on the valid patterns %%%%%
  h_input = W1' * inputs_valid + repmat(b1, 1, num_valid_cases);  % Input to hidden layer.
  h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
  logit = W2' * h_output + repmat(b2, 1, num_valid_cases);  % Input to output layer.
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
    %%%%%%%%%%%%%%%%%%%
  %For Precision fo validation
  countsaid=0;
  countfound=0;
  for i=1:size(target_valid,2)
      if(prediction(i)>0.5)
          countsaid=countsaid+1;
          if(target_valid(i))
              countfound=countfound+1;
          end
      end
  end
  precision_valid=countfound/countsaid;
  %For Recall of validation
  countthere=0;
  for i=1:size(target_valid,2)
      if(target_valid(i))
          countthere=countthere+1;
      end
  end  
  recall_valid=countfound/countthere;
  f1_valid=2*precision_valid*recall_valid/(precision_valid+recall_valid);
  %%%%%% Print out summary statistics at the end of the epoch %%%%%
  total_epochs = total_epochs + 1;
  if total_epochs == 1
      start_error = train_CE;
  end
  train_CE_list(1, epoch) = train_CE;
  valid_CE_list(1, epoch) = valid_CE;
  train_count_list(1,epoch)=countwr_train;
  valid_count_list(1,epoch)=countwr_valid;
  fprintf(1,'%d  Train CE=%f, Valid CE=%f, Train E=%f, Valid E=%f, F1 train=%f, F1 valid=%f\n',...
            total_epochs, train_CE, valid_CE, countwr_train, countwr_valid, f1_train, f1_valid);
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

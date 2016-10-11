%Code to provide classification error using the weights of the trained
%hidden units
function [CE,f1_test]= testnn_prelim(W1,W2,b1,b2)
% [inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test] = load_data();
load Encoded_umm;
  num_test_cases = size(inputs_test, 2);
  h_input = W1' * inputs_test + repmat(b1, 1, num_test_cases);  % Input to hidden layer.
  h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
  logit = W2' * h_output + repmat(b2, 1, num_test_cases);  % Input to output layer.
  prediction = 1 ./ (1 + exp(-logit));  % Output prediction.


  countwr_test=0;
  % Compute classification error for training inputs
  for i=1:size(target_test,2)
      if(target_test(i))
          if(prediction(i)<0.5)
              countwr_test=countwr_test+1;
          end
      else
          if(prediction(i)>0.5)
              countwr_test=countwr_test+1;
          end
      end
  end
  %Normalise count
  CE=countwr_test/num_test_cases;
    %%%%%%%%%%%%%%%%%%%
  %For Precision
  countsaid=0;
  countfound=0;
  for i=1:size(target_test,2)
      if(prediction(i)>0.5)
          countsaid=countsaid+1;
          if(target_test(i))
              countfound=countfound+1;
          end
      end
  end
  precision_test=countfound/countsaid;
  %For Recall
  countthere=0;
  for i=1:size(target_test,2)
      if(target_test(i))
          countthere=countthere+1;
      end
  end  
  recall_test=countfound/countthere;
  f1_test=2*precision_test*recall_test/(precision_test+recall_test);
  %%%%%%%%%%%%%%%%%%%
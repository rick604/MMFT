function [Yt_pred] = MMFT(Xs,Ys,Xt,options)
%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim

%%%% options: algorithm options:
%%%%% options.T           :  number of iteration 
%%%%% options.Kernel_type :  choose the kernel :'linear','rbf'or'sam' 
%%%%% options.lambda      :  lambda  
%%%%% options.sigma       :  sigma  
%%%%% options.idx         :  the index of source domains being transferred  

%% Outputs:
%%%% Yt_pred  :  Prediction labels for target domain

%% Algorithm starts
idx=options.idx;
X = [Xs,Xt];  
n = size(Xs,2);
m = size(Xt,2);
C = length(unique(Ys));
YY = [];
for c = 1 : C
    YY = [YY,Ys==c];
end
YY = [YY;zeros(m,C)];
X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));    
Cls=[];
    
for i=1:options.T
    FF=zeros(m*2,2); 
    for r=idx  
        XX=cat(2,X(:,1+(r-1)*m:r*m),X(:,n+1:end));
        K = kernel_meda(options.Kernel_type,XX,sqrt(sum(sum(XX.^0.5)/(m + m))));   
        w=ones(m,1);
        w(Ys(1+(r-1)*m:r*m)==2)=(sum(Ys(1+(r-1)*m:r*m)==1)/sum(Ys(1+(r-1)*m:r*m)==2));
        E = diag(sparse([w;zeros(m,1)])); 
        
        % Construct MMD matrix, only align the conditional distribution
        M = 0;
        for c = reshape(unique(Ys),1,length(unique(Ys)))
            e = zeros(m + m,1);
            e(Ys(1+(r-1)*m:r*m) == c) = 1 / length(find(Ys(1+(r-1)*m:r*m) == c));
            e(m + find(Cls == c)) = -1 / length(find(Cls == c));
            e(isinf(e)) = 0;
            M = M + e * e';
        end   
        
%         % You can try to align joint probability distributions
%         e = [1 / m * ones(m,1); -1 / m * ones(m,1)];
%         N = e * e' * length(unique(Ys));       
%         M=M+N;
        
        M = M / norm(M,'fro');
        
        % Compute  Alpha
        yyy=cat(1,YY((1+(r-1)*m:r*m),:),YY(n+1:end,:));
        Alpha = ((E + options.lambda * M ) * K + options.sigma * speye(m + m,m + m)) \ (E * yyy);
        % Quantified voting
        FF =FF+ K * Alpha; 
    end   
    [~,Cls] = max(FF,[],2);
    Cls= Cls(m+1:end);
end
Yt_pred= Cls;
end

function K = kernel_meda(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end


function K = kernel_jda(ker,X,X2,gamma)

    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K = exp(-gamma*D); 

        case 'sam'

            if isempty(X2)
                D = X'*X;
            else
                D = X'*X2;
            end
            K = exp(-gamma*acos(D).^2);

        otherwise
            error(['Unsupported kernel ' ker])
    end
end
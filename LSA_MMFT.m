function [Yt_pred] = LSA_MMFT(Xs,Ys,Xt,options)
%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim

%%%% options: algorithm options:
%%%%% options.T           :  number of iteration 
%%%%% options.Kernel_type :  choose the kernel :'linear','rbf'or'sam' 
%%%%% options.lambda      :  lambda  
%%%%% options.sigma       :  sigma  


%% Outputs:
%%%% Yt_pred  :  Prediction labels for target domain
 
%% Algorithm starts
    n = size(Xs,2);
    m = size(Xt,2);
    num=n/m;
    lebels_record=zeros(m,num+1); 
    X = [Xs,Xt];
    C = length(unique(Ys));
    YY = [];
    for c = 1 : C
        YY = [YY,Ys==c];
    end
    YY = [YY;zeros(m,C)];
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2)))); 
    Cls=[];
    
%% Construct kernel  
    for i=1:options.T
        FF=zeros(m*2,2);  
        % Domain selection after first iteration
        if i==1
            idx=1:num;
        else
            idx=IDX;
        end       
        for r=idx
            XX=cat(2,X(:,1+(r-1)*m:r*m),X(:,n+1:end));
            K = kernel_mmft(options.Kernel_type,XX,sqrt(sum(sum(XX.^0.5)/(m + m))));   
            w=ones(m,1);
            w(Ys(1+(r-1)*m:r*m)==2)=(sum(Ys(1+(r-1)*m:r*m)==1)/sum(Ys(1+(r-1)*m:r*m)==2));
            E = diag(sparse([w;zeros(m,1)])); 
            M = 0;
            for c = reshape(unique(Ys),1,length(unique(Ys)))
                e = zeros(m + m,1);
                e(Ys(1+(r-1)*m:r*m) == c) = 1 / length(find(Ys(1+(r-1)*m:r*m) == c));
                e(m + find(Cls == c)) = -1 / length(find(Cls == c));
                e(isinf(e)) = 0;
                M = M + e * e';
            end
            M = M / norm(M,'fro');       
            % Compute Alpha
            yyy=cat(1,YY((1+(r-1)*m:r*m),:),YY(n+1:end,:));
            Alpha = ((E + options.lambda * M ) * K + options.sigma * speye(m + m,m + m)) \ (E * yyy);
            FF =FF+K * Alpha;    
            
            if i==1                  
                [~,cls] = max(K * Alpha,[],2);
                lebels_record(:,r)=cls(m+1:end);     
            end
                              
        end
        [~,Cls] = max(FF,[],2);   
        Cls=Cls(m+1:end);
        
        % Domain selection
        if i==1
            cls=Cls; 
            lebels_record(:,num+1)=cls;
            
            score=zeros(1,num);
            for nn=1:num
                score(1,nn)=sum(lebels_record(:,nn)==lebels_record(:,num+1));
            end
            [~,idx]=sort(score,'ascend');
        
            mov=options.mov;
            idx=idx(1:mov);
            a=1:num;
            a(idx)=[];
            IDX=a;
        end       
    end  
    Yt_pred =Cls;
end

function K = kernel_mmft(ker,X,sigma)
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

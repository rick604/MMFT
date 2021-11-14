function [Yt_pred] = LSA_MMFT_RSVP(Xs,Ys,Xt,options)
%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim

%%%% options: algorithm options:
%%%%% options.T           :  number of iteration 
%%%%% options.Kernel_type :  choose the kernel :'linear','rbf'or'sam' 
%%%%% options.lambda      :  lambda  
%%%%% options.sigma       :  sigma  
%%%%% options.mov         :  numbers of domains want to be moved(domain selection only)
%%%%% options.trials      :  numbers of samples indifferent subjects(RSVP only)

%% Outputs:
%%%% Yt_pred  :  Prediction labels for target domain
 
%% Algorithm starts
    trials=options.trials;
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
    lebels_record=zeros(m,11);
    Cls=[];
    
%% Construct kernel
    for i=1:options.T
        Ff=zeros(m,2);
        total=0;
        % Domain selection after first iteration
        if i==1
            index=1:10;
        else
            index=IDX;
        end 
        for r=index
            num=trials(r);
            XX=cat(2,X(:,total+1:total+num),X(:,n+1:end));
            K = kernel_meda(options.Kernel_type,XX,sqrt(sum(sum(XX.^0.5)/(num + m))));   
            w=ones(num,1); 
            w(Ys(total+1:total+num)==2)=sum(Ys(total+1:total+num)==1)/sum(Ys(total+1:total+num)==2);
            E = diag(sparse([w;zeros(m,1)]));
     
            % Construct MMD matrix, only align the conditional distribution
            M = 0;
            for c = reshape(unique(Ys),1,length(unique(Ys)))
                e = zeros(num + m,1);
                e(Ys(total+1:total+num) == c) = (1 / length(find(Ys(total+1:total+num) == c)));
                e(num + find(Cls == c)) = (-1 / length(find(Cls == c)));
                e(isinf(e)) = 0;
                M = M + e * e';
            end 
            M = M / norm(M,'fro');

            % Compute Alpha
            yyy=cat(1,YY(total+1:total+num,:),YY(n+1:end,:));
            Beta = ((E + options.lambda * M ) * K + options.sigma * speye(num + m,num + m)) \ (E * yyy);
            FF = K * Beta;  
            Ff=Ff+FF(num+1:end,:);
            
            if i==1                   
                [~,cls] = max(FF(num+1:end,:),[],2);
                lebels_record(:,r)=cls;                         
            end
            total=total+num;
            
        end
        [~,Cls] = max(Ff,[],2);
        
        % Domain selection
        if i==1
            ns=10;
            cls=Cls; 
            lebels_record(:,ns+1)=cls;         
            score=zeros(1,ns);
            for nn=1:ns
                score(1,nn)=sum(lebels_record(:,nn)==lebels_record(:,ns+1));
            end
            [~,idx]=sort(score,'ascend');
        
            mov=options.mov;
            idx=idx(1:mov);
            a=1:10;
            a(idx)=[];
            IDX=a;
        end  
    end
    Yt_pred = Cls;        
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
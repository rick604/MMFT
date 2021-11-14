%% MI1-MI4 
clc;
clear;
addpath('lib')
acc=zeros(4,11);
for data_num=1:4
    dataFolder=['data\DATA',num2str(data_num),'\'];
    files=dir([dataFolder '*.mat']);    
    XRaw=[];  yAll=[]; 
    
    % Number of subjects
    % MI1-7; MI2-9; MI3-10; MI4-10.
    nSubs=length(files);
    for s=1:nSubs
        str=[dataFolder,'A',num2str(s),'.mat'];
        load(str);
        XRaw=cat(3,XRaw,X);  yAll=cat(1,yAll,y); nTrials=length(y);
    end                    
    
    fprintf('================================================\n');
    fprintf('MMFT on MI%d (%dsubjects):\n',data_num,nSubs);   
    
    % DMA for all domains
    Ca=zeros(size(XRaw,1),size(XRaw,1),size(XRaw,3));
    for f=1:nSubs
        idf=(f-1)*nTrials+1:f*nTrials;
        Ca(:,:,idf) = DMA(XRaw(:,:,idf));
    end
    
    % Each subject takes turns as the target domain
    for i=1:nSubs                  
        idt=(i-1)*nTrials+1:i*nTrials ;
        ids=1:nTrials*nSubs; ids(idt)=[];  
        Yt=yAll(idt); Ys=yAll(ids);
        XS=Logmap(Ca(:,:,ids)); 
        XT=Logmap(Ca(:,:,idt));
        
        % F-value for MI1,MI3,MI4.
        if data_num~=2
            [idx, Fs]=Fvalue(XS',Ys,size(XS,1)/5);
            XS=Fs'; XT=XT(idx,:); 
        end            

        % Grassmann manifold feature Learning
        [Xs,Xt] = GFK_Map(XS',XT',10);
        Xs=Xs'; Xt=Xt';
        
        % MMFT
        options.Kernel_type ='rbf';
        options.T=5;
        options.sigma =0.1;
        options.lambda =0.1;
        Ys(Ys==-1)=2;Yt(Yt==-1)=2;       
        % Transfer all source domains 
        options.idx=1:nSubs-1;
        Pred=MMFT(Xs,Ys,Xt,options);
        Acc=100*mean(Pred==Yt);
        acc(data_num,i)=Acc; 
        
%         % MEKT,for comparison, use labels 0,1.
%         options.d = 10;             % subspace bases 
%         options.T = 5;              % iterations, default=5
%         options.alpha= 0.01;        % the parameter for source discriminability
%         options.beta = 0.1;         % the parameter for target locality, default=0.1
%         options.rho = 20;           % the parameter for subspace discrepancy
%         options.clf = 'slda';        % the string for base classifier, 'slda' or 'svm'
%         Cls = [];
%         [Zs, Zt] = MEKT(XS, XT, Ys, Cls, options);
%         yPred = slda(Zt,Zs,Ys);
%         Acc=100*mean(yPred==Yt);
%         acc(data_num,i)=Acc; 
        

        fprintf('Subject%d as target domain, BCA=%.2f\n',i,Acc) ;        
    end 
    BCA=mean(acc(data_num,1:nSubs));
    acc(data_num,end)=BCA;
    fprintf('Mean BCA using MMFT on MI%d: %.2f\n',data_num,BCA);
    fprintf('\n');   
end
   

 





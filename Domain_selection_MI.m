%% Domain selection for MI,including two comparison methods ROD and DTE 
clc;
clear;
addpath('lib')
% MI1-MI4
for data_num=1:4       
    dataFolder=['data\DATA',num2str(data_num),'\'];
    files=dir([dataFolder '*.mat']);    
    XRaw=[];  yAll=[]; nSubs=length(files);
    for s=1:nSubs
        str=[dataFolder,'A',num2str(s),'.mat'];
        load(str);
        XRaw=cat(3,XRaw,X);  yAll=cat(1,yAll,y); nTrials=length(y);
    end
    
    fprintf('================================================\n');
    fprintf('Domain selection on MI%d (%dsubjects):\n',data_num,nSubs);     
    
    % Mean of classification accuracy as the number of 
    % source domains gradually decreased to only one
    result=zeros(1,nSubs-1);                     
    
    % DMA
    Ca=zeros(size(XRaw,1),size(XRaw,1),size(XRaw,3));
    for f=1:nSubs
        idf=(f-1)*nTrials+1:f*nTrials;
        Ca(:,:,idf) = DMA(XRaw(:,:,idf));
    end
       
    % Number of source domains being removed 
    for mov=0:nSubs-2   
    acc=zeros(1,nSubs+1);
    for n=1:nSubs
        idt=(n-1)*nTrials+1:n*nTrials ;
        ids=1:nTrials*nSubs; ids(idt)=[];  
        Yt=yAll(idt); Ys=yAll(ids);
        XS=logmap(Ca(:,:,ids),'MI'); 
        XT=logmap(Ca(:,:,idt),'MI');
        if data_num~=2
            [idx, Fs]=Fvalue(XS',Ys,size(XS,1)/5);
            XS=Fs'; XT=XT(idx,:); 
        end
        
        % Grassmann manifold feature Learning
        [Xs,Xt] = GFK_Map(XS',XT',10);
        Xs=Xs'; Xt=Xt';

        % MMFT setting
        options.Kernel_type ='rbf';
        options.T=5;
        options.sigma =0.1;
        options.lambda =0.1;
        options.mov=mov;
        Ys(Ys==-1)=2;Yt(Yt==-1)=2;  
        
        
%         % ROD
%         rk=nan(nSubs-1,1);
%         for te=1:nSubs-1
%             idS=nTrials*(te-1)+1:nTrials*te;
%             rk(te)=RODKL(XS(:,idS)',XT',100);
%         end
%         idx = [(1:nSubs-1)',rk];
%         idx = flip(sortrows(idx,2),1);
%         IDX_ROD = sort(idx(1:end-mov,1));
%         options.idx=IDX_ROD';       
%         Pred=MMFT(Xs',Ys,Xt',options);        


%         % DTE
%         rk=nan(2,nSubs-1);
%         for te=1:nSubs-1
%             idS=nTrials*(te-1)+1:nTrials*te;
%             rk(:,te)=DTE(XS(:,idS)',XT',Ys(idS));
%         end
%         rk(1,:)=mapminmax(rk(1,:),1,0);
%         rk(2,:)=mapminmax(rk(2,:),0,1);
%         a=rk(1,:).*rk(2,:);
%         [~,index] = sort(a,'descend');
%         IDX_DTE = index(1:end-mov);  
%         options.idx=IDX_DTE;
%         Pred=MMFT(Xs',Ys,Xt',options);        
        
                
        % LSA-MMFT 
        Pred=LSA_MMFT(Xs,Ys,Xt,options);
        
        Acc=mean(Pred==Yt)*100;
        acc(1,n)=Acc;
    end  
        acc(1,end)=mean(acc(1:n));
        result(1,mov+1)=mean(acc(1:n));
        fprintf('Remove %d source domains, mean accuracy=%.2f\n',mov,mean(acc(1:n)))  
    end 
   result=result';
end







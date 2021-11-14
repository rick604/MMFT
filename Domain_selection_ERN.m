%% Domain selection for ERN,including two comparison methods ROD and DTE
clc;
clear ;
% 16 subjects, each 56*260*340 (channels*points*trails)
root='data\ERN\';
listing=dir([root '*.mat']);
nSubs=length(listing);
addpath('lib')
nTrials=340;
Xc=zeros(56,260,340*nSubs);
Xr=zeros(56,260,340*nSubs);
Y=nan(340*nSubs,1);
for f=1:nSubs
    load([root listing(f).name])
    idf=(f-1)*340+1:f*340;
    Y(idf)=y;
    Xr(:,:,idf)=x;
    [~,Xc(:,:,idf)]=DMA_E(x);
end

fprintf('================================================\n');
fprintf('Domain selection on ERN (%dsubjects):\n',nSubs);   

BCA=zeros(1,nSubs+1);
bca=zeros(8,17);
result=zeros(8,1);
count=1;
% Number of source domains being removed
for mov=0:2:14    
    for n=1:nSubs
        % Single target data   
        idt=(n-1)*340+1:n*340;
        ids=1:340*nSubs; ids(idt)=[];          
        
        % Multi source data
        Xsc=Xc(:,:,ids); Xtc=Xc(:,:,idt);
        Ys=Y(ids); Yt=Y(idt);
        idsP=Yt==1; idsN=Yt==0;
        w=ones(size(Ys)); w(Ys==1)=sum(Ys==0)/sum(Ys==1);
        
        % xDAWN filtering
        [xTrain,xTest] = xDAWN(5,Xsc,Ys,Xtc);
        % Compute SCM by the raw source data
        E=mean(xTrain(:,:,Ys==1),3); 
        Xsn=cat(1,repmat(E,[1,1,length(Ys)]),xTrain);
        Xtn=cat(1,repmat(E,[1,1,length(Yt)]),xTest);         
        for f=1:nSubs-1
            idf=(f-1)*340+1:f*340;
            [Ca(:,:,idf)]=DMA(Xsn(:,:,idf));
        end 
        Cs=Ca;
        Ct=DMA(Xtn);
        
        % Logarithmic mapping on aligned covariance matrices
        XS=Logmap(Cs);
        XT=Logmap(Ct); 

        % Grassmann manifold feature Learning
        [Xs,Xt] = GFK_Map(XS',XT',10);
        Xs=Xs'; Xt=Xt';
        
        % MMFT setting
        options.Kernel_type ='rbf';
        options.sigma =0.1;
        options.lambda =0.1;   
        options.T=5;
        Ys(Ys==0)=2;Yt(Yt==0)=2;
        options.mov=mov;
        
        
%         % ROD       
%         rk=nan(nSubs-1,1);
%         for te=1:nSubs-1
%             idS=nTrials*(te-1)+1:nTrials*te;
%             rk(te)=RODKL(XS(:,idS)',XT',20);
%         end
%         idx = [(1:nSubs-1)',rk];
%         idx = flip(sortrows(idx,2),1);
%         IDX_ROD = sort(idx(1:end-mov,1));
%         options.idx=IDX_ROD';
%         Ypre=MMFT(Xs,Ys,Xt,options);
       
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
%         Ypre=MMFT(Xs,Ys,Xt,options);
        

        % LSA-MMFT
        Ypre = LSA_MMFT(Xs,Ys,Xt,options);


        bca=.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==2))*100;
        BCA(1,n)= bca;  

    end
    BCA(1,end)=mean(BCA(1:nSubs));   
    fprintf('Remove %d source domains, mean accuracy=%.2f\n',mov,mean(BCA(1:16))) ;
    result(count)=mean(BCA(1:16));
    count=count+1;
end




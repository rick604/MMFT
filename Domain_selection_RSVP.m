%% Domain selection for RSVP,including two comparison methods ROD and DTE
clc;
clear;
% Load datasets: 
% 11 subjects, each 8*45*n (channels*points*trails)
load('data\RSVP.mat')
addpath('lib')
nSubs=11;
% Load data and perform congruent transform
%nTrials=11,11 subjects
fnum=length(nTrials);   
[m,n,~]=size(xAll);
Xc=zeros(m,n,length(yAll));

for t=1:fnum
    idx=sum(nTrials(1:t-1));
    idf=idx+1:idx+nTrials(t);
    xr=xAll(:,:,idf); yr=yAll(idf);
    [~,Xc(:,:,idf)]=DMA_E(xr);
end

count=1;
result=zeros(10,1);
% Number of source domains being removed
for mov=0:9
    BCA=zeros(1,fnum+1);
    for n=1:fnum
        Ca=[];
        trials=nTrials;
        trials(n)=[];
        % Single target data
        idx=sum(nTrials(1:n-1));
        idt=idx+1:idx+nTrials(n);
        ids=1:length(yAll);
        ids(idt)=[];

        % Multi source data
        Xsc=Xc(:,:,ids); Xtc=Xc(:,:,idt);
        Ys=yAll(ids); Yt=yAll(idt);
        idsP=Yt==1; idsN=Yt==0;

        % xDAWN filtering       
        [xTrain,xTest]=xDAWN(3,Xsc,Ys,Xtc);
        E=mean(xTrain(:,:,Ys==1),3);  % Compute SCM by the raw source data
        Xsn=cat(1,repmat(E,[1,1,length(Ys)]),xTrain);
        Xtn=cat(1,repmat(E,[1,1,length(Yt)]),xTest);

        temptrial=nTrials;
        temptrial(n)=[];
        idx=0;
        for f=1:fnum-1            
            idff=1+idx:idx+temptrial(f);
            [Ca(:,:,idff)]=DMA(Xsn(:,:,idff));
            idx=idx+temptrial(f);
        end  
        Cs=Ca;
        Ct=DMA(Xtn);
                          
        % Logarithmic mapping on aligned covariance matrices
        XS=Logmap(Cs);
        XT=Logmap(Ct); 

        % Grassmann manifold feature Learning
        [Xs,Xt] = GFK_Map(XS',XT',10);
        Xs=Xs'; Xt=Xt';
        
        options.Kernel_type ='rbf'; 
        options.sigma =0.1;
        options.lambda =0.1; 
        options.T=5;
        options.mov=mov;
        options.trials=trials;
        Ys(Ys==0)=2;Yt(Yt==0)=2;  
        
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
%         Ypre=MMFT_RSVP(Xs,Ys,Xt,options);


        % DTE
        rk=nan(2,nSubs-1);
        for te=1:nSubs-1
            idS=nTrials*(te-1)+1:nTrials*te;
            rk(:,te)=DTE(XS(:,idS)',XT',Ys(idS));
        end
        rk(1,:)=mapminmax(rk(1,:),1,0);
        rk(2,:)=mapminmax(rk(2,:),0,1);
        a=rk(1,:).*rk(2,:);
        [~,index] = sort(a,'descend');
        IDX_DTE = index(1:end-mov); 
        options.idx=IDX_DTE';
        Ypre=MMFT_RSVP(Xs,Ys,Xt,options);   


%         % MMFT  
%         Ypre=LSA_MMFT_RSVP(Xs,Ys,Xt,options);
        
        
        bca=0.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==2))*100;
        BCA(1,n)=bca;
    end
    BCA(1,end)=mean(BCA(1:fnum));
    fprintf('Remove %d source domains, mean accuracy=%.2f\n',mov,mean(BCA(1:11))) ;
    result(count)=mean(BCA(1:11));
    count=count+1;
end


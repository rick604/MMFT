clc;
clear;

% 11 subjects, each 8*45*n (channels*points*trails)
load('data\RSVP.mat')
addpath('lib')
nSubs=length(nTrials);   
[m,n,~]=size(xAll);
Xc=zeros(m,n,length(yAll));

for t=1:nSubs
    idx=sum(nTrials(1:t-1));
    idf=idx+1:idx+nTrials(t);
    xr=xAll(:,:,idf); yr=yAll(idf);
    [~,Xc(:,:,idf)]=DMA_E(xr);
end
fprintf('================================================\n');
fprintf('MMFT on RSVP dataset (%dsubjects):\n',nSubs);
BCA=zeros(1,nSubs+1);
for n=1:nSubs
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
    
    % DMA for all domains
    temptrial=nTrials;
    temptrial(n)=[];
    idx=0;
    for f=1:nSubs-1            
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
    
    % MMFT
    options.Kernel_type ='rbf';
    options.T=5;
    options.sigma =0.1;
    options.lambda =0.1; 
    options.trials=trials;
    options.idx=1:10;
    Ys(Ys==0)=2;Yt(Yt==0)=2;      
    % The dimensions of different source domains are inconsistent. Modify the MMFT slightly
    Ypre=MMFT_RSVP(Xs,Ys,Xt,options);
    bca=0.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==2))*100;


%     % MEKT,for comparison, use labels 0,1.
%     options.d = 10;             % subspace bases 
%     options.T = 5;              % iterations, default=5
%     options.alpha= 0.01;        % the parameter for source discriminability
%     options.beta = 0.1;         % the parameter for target locality, default=0.1
%     options.rho = 20;           % the parameter for subspace discrepancy
%     options.clf = 'svm';        % the string for base classifier, 'slda' or 'svm'
%     Cls = [];
%     [Zs, Zt] = MEKT(XS, XT, Ys, Cls, options);
%     w=ones(size(Ys)); w(Ys==1)=sum(Ys==0)/sum(Ys==1);
%     model = libsvmtrain(w,Ys,Zs','-h 0 -t 0 -c 0.125');
%     Ypre = libpredict(Yt,Zt',model);
%     bca=0.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==0))*100;

    BCA(1,n)= bca;  
    fprintf('Subject%d as target domain, BCA=%.2f\n',n,bca) ;
end
BCA(1,end)=mean(BCA(1:nSubs));
meanbca=BCA(1,end);
fprintf('Mean BCA using MMFT on RSVP: %.2f\n',meanbca);
fprintf('\n'); 



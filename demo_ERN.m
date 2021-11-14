%% ERN dataset£¬16 subjects, each 56*260*340 (channels*points*trails)
clc;
clear;
root='data\ERN\';
listing=dir([root '*.mat']);
nSubs=length(listing);
addpath('lib')
Xc=zeros(56,260,340*nSubs);
Xr=zeros(56,260,340*nSubs);
Y=nan(340*nSubs,1);

for f=1:nSubs
    load([root listing(f).name])
    idf=(f-1)*340+1:f*340;
    Y(idf)=y;
    Xr(:,:,idf)=x;
    % DMA on Euclidean space
    [~,Xc(:,:,idf)]=DMA_E(x);
end
fprintf('================================================\n');
fprintf('MMFT on ERN dataset (%dsubjects):\n',nSubs);
BCA=zeros(1,nSubs+1);        
for n=1:nSubs
    % Single target data   
    idt=(n-1)*340+1:n*340;
    ids=1:340*nSubs; ids(idt)=[];          

    % Multi source data
    Xsc=Xc(:,:,ids); Xtc=Xc(:,:,idt);
    Ys=Y(ids); Yt=Y(idt);
    % For BCA calculation
    idsP=Yt==1; idsN=Yt==0;
        
    % xDAWN filtering
    [xTrain,xTest] = xDAWN(5,Xsc,Ys,Xtc);
    
    % Compute SCM matrices, 1 is the target class 
    E=mean(xTrain(:,:,Ys==1),3); 
    Xsn=cat(1,repmat(E,[1,1,length(Ys)]),xTrain);
    Xtn=cat(1,repmat(E,[1,1,length(Yt)]),xTest);   
                    
    % DMA for all domains
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
        
    % MMFT
    options.Kernel_type ='rbf';
    options.T=5;
    options.sigma =0.1;
    options.lambda =0.1;     
    Ys(Ys==0)=2;Yt(Yt==0)=2;
    options.idxs=1:15;
    [Ypre] = MMFT(Xs,Ys,Xt,options);
    bca=.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==2))*100;

       
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
fprintf('Mean BCA using MMFT on ERN: %.2f\n',meanbca);
fprintf('\n');     





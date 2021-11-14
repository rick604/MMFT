function C=mean_covariances(P,str)
C=zeros(size(P,1),size(P,1));
if strcmp(str,'riemann')
    C=riemann_mean(P);

              
elseif strcmp(str,'logeuclid') 
    for i=1:size(P,3)
        C=C+(size(P,3)^(-1))*logm(P(:,:,i));
    end
    C=expm(C);

elseif strcmp(str,'euclid')
    for i=1:size(P,3)
        C=C+P(:,:,i);
    end
    C=C/size(P,3);
          
end

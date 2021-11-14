function Fea = Logmap(COV)
% Logarithmic mapping on aligned covariance matrices
% Input:
%   COV: c*c*N, centralized signal covariance matrices
% Output:
%   Fea: tangent space features, d*N
    
NTrial = size(COV,3);
N_elec = size(COV,1);

% Select upper triangular elements related to spatial information
Fea = zeros(N_elec*(N_elec+1)/2,NTrial);
index = reshape(triu(ones(N_elec)),N_elec*N_elec,1)==1;
for i=1:NTrial
    Tn = logm(COV(:,:,i));
    tmp = reshape(sqrt(2)*triu(Tn,1)+diag(diag(Tn)),N_elec*N_elec,1);
    Fea(:,i) = tmp(index);
end

function Q=buildQ(Theta,Wi,T,P,N,S)
% Build the matrix 'Q'

Wbar=repmat(reshape(Wi,[T*P 1]),[N 1]); % [TPN x 1] 
Qcell=cell(1,S);
for s=1:S
	Qcell{s}=reshape(sparse(Theta{s}*Wbar),[T P*N]); % [T x PN]
end
Q=blkdiag(Qcell{:}); % [TS x PNS]

end %#EoF buildQ
function VAF=getVAF(Mb,E2,S)

T=size(Mb,1)/S;
M=size(Mb,2);

% Get mean muscle pattern
Mmean=zeros(T,M);
for s=1:S 
	Mmean=Mmean+Mb(T*(s-1)+1:T*s,:)/S; 
end


% Total Variance
SST=0;
for s=1:S
    Mb0=Mb(T*(s-1)+1:T*s,:)-Mmean;
    SST=SST+norm(Mb0,'fro')^2;
end

% Variance Accounted For
VAF=1-E2/SST;

end %#EoF getVAF

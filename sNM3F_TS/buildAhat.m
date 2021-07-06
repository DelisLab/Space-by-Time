function Ahat=buildAhat(A,P,N,S)
% Build the Ahat matrix by creating a large block-diagonal matrix. 
% Each block refers to a given sample s, itself composed of a block 
% diagonal matrix built from the columns of As

Ahat=zeros(P*N*S,N);
for s=1:S 
    AhatBlock=zeros(P*N,N);
    for j=1:N
        AhatBlock((j-1)*P+1:j*P,j)=A((s-1)*P+1:s*P,j);
    end
    Ahat((s-1)*P*N+1:s*P*N,:)=AhatBlock;
end
Ahat=sparse(Ahat);

end %#EoF buildAhat 
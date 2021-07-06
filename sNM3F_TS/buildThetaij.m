function Thetaij=buildThetaij(TS,T,P,N,S)
% Build the matrices 'Thetaij' 

Thetaij=cell(T,P);
for i=1:T
    for j=1:P
         
        %- Vectorized version for maximal speed
        % Preliminary steps to build Thetaij{i,j} in a vectorized fashion
        p=j:P:j+(N-1)*P;
        s=1:S;
        k=i+TS(p,s); % if k>T or k<1 we will put NaNs
        m=(k>T)+(k<1);
        [r,c]=find(m);
        for idx=1:length(r), k(r(idx),c(idx))=NaN; end
        ktrue = bsxfun(@plus,k,repmat((s-1)*T,size(k,1),1));    
        kvect=ktrue(:);  
        idx=isnan(kvect);
        kk=kvect(~idx);
        pmat=repmat(p',1,S);
        ptrue = bsxfun(@plus,pmat,repmat((s-1)*P*N,size(pmat,1),1));      
        pvect=ptrue(:);
        pp=pvect(~idx);
        % Return the sparse matrix
        Thetaij{i,j}=sparse(kk,pp,ones(size(kk)),T*S,P*N*S);
        
        %- Alternative, more explicit but less efficient implementation
%         Thetaijtmp=cell(1,S); % subdiag k=i+d, upp diag i=k-d
%         for s=1:S
%             Ttmp=zeros(T,P*N); 
%             for n=1:N
%                 p=j+(n-1)*P;
%                 d=TS(p,s);
%                 k=i+d;
%                 if k<=T && k>=1
%                   Ttmp(k,p)=1;
%                 end                 
%             end
%             Thetaijtmp{s}=sparse(Ttmp);
%         end
%                 
%         Thetaij{i,j}=blkdiag(Thetaijtmp{:});

    end
end

end %#EoF buildThetaij
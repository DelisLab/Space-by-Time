function [val, gradient] = nlpCostFun(z,p)

gamma=p.gamma;
delta=p.delta;

skip=0;
if gamma==0 && delta==0,
    skip=1;
end

Mb=p.Mb;
S=p.S;
N=p.N;
P=p.P;
MSC=p.MSC;
T=p.T;
G=p.G;
NG=p.NG;

Wi=reshape(z(1:T*P),[T P]);
Wb=reshape(z(end-MSC*N+1:end),[N MSC]);
Avect=zeros(P*N,S);

%-------------------------------------- NMF-based cost
if (nargout > 1)
    gradientMat_Wi=zeros(T,P);
    gradientMat_As=zeros(P,N*S);
    gradientMat_Wb=zeros(N,MSC);
end

errNMF=0;
for s=1:S
   As=reshape(z(T*P+P*N*(s-1)+1:T*P+P*N*s),[P,N]); 
   Avect(:,s)=reshape(As,[P*N,1]); % P elements first (for N=1), P elements then (for N=2) etc.
   errNMF=errNMF+norm(Mb((s-1)*T+1:s*T,:)-Wi*As*Wb,'fro')^2;  
   
   if (nargout > 1)
    % Formulae from e.g. the matrix CookBook   
	gradientMat_Wi = gradientMat_Wi + 2*(Wi*As*Wb-Mb((s-1)*T+1:s*T,:))*(As*Wb)' ;
    gradientMat_As(:,(s-1)*N+1:s*N) = 2*Wi'*(Wi*As*Wb-Mb((s-1)*T+1:s*T,:))*Wb' ;
    gradientMat_Wb = gradientMat_Wb + 2*(Wi*As)'*(Wi*As*Wb-Mb((s-1)*T+1:s*T,:)) ;
   end
end

% Gradient for Frobenius norm
if (nargout > 1)
    gradientNMF=zeros(numel(z),1);
    gradientNMF(1:T*P)=reshape(gradientMat_Wi,[T*P 1]);
    gradientNMF(T*P+1:T*P+P*N*S)=reshape(gradientMat_As,[P*N*S 1]);
    gradientNMF(end-MSC*N+1:end)=reshape(gradientMat_Wb,[N*MSC 1]);
end

%-------------------------------------- LDA-based cost

% Computation of within and between class scatter matrices
Sb=0;
Sw=0;
AmeanGlobal=mean(Avect,2); % global mean
if (nargout > 1)
    AmeanGlobalGradient=1/S*repmat(eye(N*P), [1 S]);
    gradientLDATmp=zeros(N*P*S,1);
end

sg=zeros(NG,S/NG);
Amean=zeros(P*N,NG);

if ~skip, % we skip these calculations if gamma=0 and delta=0
    for i=1:NG
       sg(i,:)=find(G==i); % trials belonging to group i
       NGi=numel(sg(i,:));
       % mean A vector for each group
       Amean(:,i)=mean(Avect(:,sg(i,:)),2);

       Sb=Sb+(Amean(:,i)-AmeanGlobal)'*(Amean(:,i)-AmeanGlobal);

       if (nargout > 1)
           AmeanGradient=zeros(P*N,P*N*S);  % one matrix for each i
       end

       for si=1:NGi 
           if (nargout > 1)    
              for j=1:N*P            
                 col=(sg(i,si)-1)*P*N+j;
                 AmeanGradient(mod(col-1,N*P)+1,col)=1/NGi;    
              end  
           end   
           Sw=Sw+(Avect(:,sg(i,si))-Amean(:,i))'*(Avect(:,sg(i,si))-Amean(:,i));
       end

       if (nargout > 1)
           % Gradient for Sw (within-class scatter matrix) 
           for si=1:NGi
              AvectGradient=zeros(P*N,P*N*S); % one matrix for each i, si
              for j=1:N*P  
                 col=(sg(i,si)-1)*P*N+j; 
                 AvectGradient(mod(col-1,N*P)+1,col)=1;    
              end
              dAvectAmeanGradient = AvectGradient - AmeanGradient;
              gradientLDATmp=gradientLDATmp+gamma*2*dAvectAmeanGradient'*(Avect(:,sg(i,si))-Amean(:,i));
           end

           % Gradient for Sb (between-class scatter matrix)
           dAmeanAmeanGobalGradient = AmeanGradient - AmeanGlobalGradient;
           gradientLDATmp=gradientLDATmp-delta*2*dAmeanAmeanGobalGradient'*(Amean(:,i)-AmeanGlobal);
       end
    end
end

if skip,
    errLDA=0;
else
    errLDA=gamma*Sw-delta*Sb;
end


if (nargout > 1)
    gradientLDA=zeros(numel(z),1);
    if ~skip,
        gradientLDA(1:T*P)=0; % independent of Wi
        gradientLDA(end-MSC*N+1:end)=0; % independent of Wb
        gradientLDA(T*P+1:T*P+P*N*S) = gradientLDATmp;    
    end
end


% Total error
val=errNMF+errLDA;

if (nargout > 1)
    gradient = gradientLDA + gradientNMF ;
end

end %#EoF nlpCostFun

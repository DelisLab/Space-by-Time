function [c,ceq,Gc,Gceq] = nlpConstFun(z,p)
    
N=p.N;
P=p.P;
MSC=p.MSC;
T=p.T;
S=p.S;

c=[];
Gc=[];

Wi=reshape(z(1:T*P),[T P]);
Wb=reshape(z(end-MSC*N+1:end),[N MSC]);

ceq=zeros(1,N+P);
for i=1:P
   ceq(i)=sum(Wi(:,i))-1;
end

for j=1:N
    ceq(P+j)=sum(Wb(j,:))-1;
end

if (nargout > 2)

    Gceq=zeros(numel(z),N+P);

    for i=1:P
       gradtmp=ones(T,1);
       Gceq((i-1)*T+1:i*T,i)=gradtmp;
    end
    for j=1:N
       gradtmp=ones(MSC,1);
       ini=T*P+N*P*S;
       for k=1:MSC   
        Gceq(ini+(k-1)*N+j,P+j)=gradtmp(k);
       end
       
    end

end %#EoF nlpConstFun
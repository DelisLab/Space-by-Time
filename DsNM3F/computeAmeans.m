function [AmeanAll,Amean,Sb] = computeAmeans(A,P,N,G)

if nargout>2,
 Sb=0;
end

% Compute AmeanAll and Amean for each class
AmeanAll=zeros(P,N);
for j=1:N
   AmeanAll(:,j)=mean(A(:,j:N:end),2);
end 
NG=max(G); % maximum element of G is the number of groups

Amean=zeros(P,N*NG);
for i=1:NG  
   sGi=find(G==i); % trials belonging to group i
   NGi=numel(sGi);
   for si=1:NGi
    sgi=sGi(si); 
    Amean(:,(i-1)*N+1:i*N)=Amean(:,(i-1)*N+1:i*N)+1/NGi*A(:,N*(sgi-1)+1:N*sgi); 
   end
   if nargout>2,
     Sb=Sb+trace((Amean(:,(i-1)*N+1:i*N)-AmeanAll)'*(Amean(:,(i-1)*N+1:i*N)-AmeanAll));
   end  
end

end %#EoF computeAmeans


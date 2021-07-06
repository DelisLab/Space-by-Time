function [Anew,Wi,Wb,TSnew] = cleanOutput(A,Wi,Wb,TS,P,N,S,T)
% Re-order and normalize the output

% Order elements by the occurence of the maximal element
Anew=zeros(size(A));

% Column-wise for Wi
iP=zeros(1,P);
sP=zeros(1,P); % re-order normalization
for i=1:P
    [~,iP(i)]=max(Wi(:,i));
    sP(i)=norm(Wi(:,i));
    Wi(:,i)=Wi(:,i)./sP(i);
    % Try to shift temporal modules leftward to prevent abusive positive delay
    dmin=min(min(TS(i:P:end,:),[],1),[],2); %
    if dmin>0
    	Wishifted=zeros(T,1); % initialize to zero
    	Wishifted(dmin+1:end,1)=Wi(1:T-dmin,i); % time-shift the selected temporal module by the computed delay                    
    	Wi(:,i)=Wishifted;
    	TS(i:P:end,:)=TS(i:P:end,:)-dmin;
    end 
    % Try to shift temporal modules rightward to prevent abusive negative delay
    dmax=max(max(TS(i:P:end,:),[],1),[],2); %
    if dmax<0
    	Wishifted=zeros(T,1); % initialize to zero
    	Wishifted(1:T+dmax,1)=Wi(-dmax+1:end,i); % time-shift the selected temporal module by the computed delay                    
    	Wi(:,i)=Wishifted;
    	TS(i:P:end,:)=TS(i:P:end,:)-dmax;
    end 
end


[~,idxP] = sort(iP,'ascend');

% Row-wise for Wb
iN=zeros(1,N);
sN=zeros(1,N); % re-order normalization
for j=1:N
    [~,iN(j)]=max(Wb(j,:));
    sN(j)=norm(Wb(j,:));
    Wb(j,:)=Wb(j,:)./sN(j);
end


[~,idxN] = sort(iN,'ascend');

% Re-order all other elements according to the new order of modules
Wi=Wi(:,idxP);
Wb=Wb(idxN,:);
TSnew=zeros(size(TS));
for j=1:N
    for i=1:P
        TSnew((j-1)*P+i,:)=TS((idxN(j)-1)*P+idxP(i),:);
    end
end
for s=1:S
    for i=1:P
        for j=1:N
            % Element-wise normalization of A to make the error unchanged
            Anew(P*(s-1)+i,j)=A(P*(s-1)+idxP(i),idxN(j)).*(sP(idxP(i))*sN(idxN(j))); 
        end
    end
end

end %#EoF cleanOutput
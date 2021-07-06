function [errNMF,errLDAw,errLDAb] = computeError(Mi,Wi,A,Wb,gamma,delta,P,N,S,G)

errNMF=0;
if nargout>1,
    Sw=0;
    [~,Amean,Sb] = computeAmeans(A,P,N,G);
end

M=size(Wb,2);
for s=1:S
    if nargout>1,
        gs=G(s); % group of trial s
        Sw=Sw+trace((A(:,N*(s-1)+1:N*s)-Amean(:,(gs-1)*N+1:gs*N))'*(A(:,N*(s-1)+1:N*s)-Amean(:,(gs-1)*N+1:gs*N)));
    end
    errNMF=errNMF+norm(Mi(:,M*(s-1)+1:M*s)-Wi*A(:,N*(s-1)+1:N*s)*Wb,'fro')^2;
end

if nargout>1,
    errLDAw=gamma*Sw;
    errLDAb=-delta*Sb;
end

% NB: the total cost function is errNMF+errLDAw+errLDAb

end %#EoF computeError


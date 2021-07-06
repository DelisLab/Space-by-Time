function [Theta,Thetaij]=buildTheta(TS,P,N,S,D)
% Builds the very large yet sparse matrix 'Theta' to handle time-shifts
% Handles negative time-shifts

Theta=cell(1,S);
for s=1:S
    Thetas=cell(1,P*N);
    for p=1:P*N
         d=TS(p,s);
         if d>=0,
            Thetas{p}=D{d+1};
         else
            Thetas{p}=D{-d+1}';
         end
    end
    Theta{s}=blkdiag(Thetas{:});
end

% Computation of Thetaij from Theta. This implementation is likely to be 
% slow with large datasets but the code is more readable) - an optimized 
% version is implemented in 'buildThetaij.m' and should be preferred for
% speed performance
if nargout==2
    T=size(D{1},1);
    Thetaij=cell(T,P); % since i=1..T and j=1..P
    ctmp=cell(S,T,P);
    for s=1:S
        for i=1:T
            for j=1:P
                Thetaijtmp=sparse(T,P*N); 
                for n=1:N                   
                  Thetaijtmp=Thetaijtmp+reshape(Theta{s}(:,(j-1)*T+i+(n-1)*T*P),[T P*N]);
                end
                ctmp{s,i,j}=Thetaijtmp;
            end    
        end
    end
    for i=1:T
        for j=1:P
            Thetaij{i,j}=sparse(blkdiag(ctmp{:,i,j}));
        end
    end   
end


end %#EoF buildTheta
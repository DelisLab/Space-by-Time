function TS=updateTSfast(TS,Mb,A,Wi,Wb,T,M,P,N,S,MAXTS,MINTS)
% Update the Time-Shifts via a cross-correlation analysis
% Note that this is a basic implementation with only two for loops
% A better, but slower, implementation can be found in updateTS.m 
%------------------------------------------------------------------------
% N.B.: this function may drastically slow down the program with large
% datasets. Create a compiled mex file to speed up (e.g. using codegen)
% Contact the authors for more information
%------------------------------------------------------------------------

for s=1:S % for all samples (i.e. trials)         	
    Mbs=Mb((s-1)*T+1:s*T,:); % muscle signals for sample s  
    for i=P:-1:1 % backward for the temporal modules               
        for j=1:N 
            k=(j-1)*P+i;
            d=TS(k,s); % TS for this temporal/spatial modules pair
            if d>=0
                Wishifted=zeros(T,1);% time-shifted temporal module
                Wishifted(d+1:end,:)=Wi(1:T-d,i); 
            else
                Wishifted=zeros(T,1);% time-shifted temporal module
                Wishifted(1:T+d,:)=Wi(-d+1:end,i); 
            end
            % Compute all possible cross-correlations
            corrl2=zeros(2*T-1,M);
            lag=-T+1:T-1;
            Wbj=Wb(j,:);                  
            for m=1:M
               tmp=Wishifted*A((s-1)*P+i,j)*Wbj;
               corrl2(:,m)=xcorr(Mbs(:,m),tmp(:,m)); % may add ,'coeff'
            end
            corrl=sum(corrl2,2);         
            lags=lag(T+MINTS-d:T+MAXTS-d);
            % Store the best xcorr for the current pair of modules     
            [~, bestlag]=max(corrl(T+MINTS-d:T+MAXTS-d));
            % Update the TS for the current pair of modules
            TS(k,s)=d+lags(bestlag);  
            % Compute time-shifted version of selected temporal module
            newd=TS(k,s);
            if newd>=0
                Wishifted=zeros(T,1);% positively time-shifted module
                Wishifted(newd+1:end,:)=Wi(1:T-newd,i);
            else
                Wishifted=zeros(T,1);% negatively time-shifted module
                Wishifted(1:T+newd,:)=Wi(-newd+1:end,i); 
            end           
            % Subtract the selected pair of spatial-temporal modules 
            % from the input matrix Mbs
            Mbs=Mbs-Wishifted*A((s-1)*P+i,j)*Wb(j,:); 
        end
    end
end

end %#EoF updateTSfast
function TS=updateTS(TS,Mb,A,Wi,Wb,T,M,P,N,S,MAXTS,MINTS)
% Update the Time-Shifts via a cross-correlation analysis
%------------------------------------------------------------------------
% N.B.: this function may drastically slow down the program 
% Consider compiling this function to speed up (e.g. using codegen)
% Contact the authors for more information
%------------------------------------------------------------------------

for s=1:S % for all samples (i.e. trials)         	
    Mbs=Mb((s-1)*T+1:s*T,:); % muscle signals for sample s  
    mod_done=NaN(P,N); % Array to store the already treated modules   
    rn=length(mod_done(~isnan(mod_done)));
    while rn<P*N % stop when the procedure has been done for all modules 
        % Initialize some variables
        corr_tmp=zeros(1,P*N-rn); TS_tmp=NaN(1,P*N-rn); 
        id_spatmod=NaN(1,P*N-rn); id_tempmod=NaN(1,P*N-rn);  
        cp=0; % counter
        for i=1:P               
            for j=1:N % a priori, we do it for all modules 
                      % but see the "if" hereafter: we only treat the 
                      % remaining modules      
                if isnan(mod_done(i,j)) 
                    cp=cp+1; k=(j-1)*P+i;
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
                    [bestcorr, bestlag]=max(corrl(T+MINTS-d:T+MAXTS-d));
                    % Compute the best TS for the current pair of modules
                    TS_tmp(cp)=d+lags(bestlag);
                    corr_tmp(cp)=bestcorr;
                    id_spatmod(cp)=j;
                    id_tempmod(cp)=i;
                end
            end
        end
        % Select the pair of modules with highest xcorr
        [~,bestmodind]=max(corr_tmp);
        jbest=id_spatmod(bestmodind); 
        ibest=id_tempmod(bestmodind); 
        kbest=(jbest-1)*P+ibest;
        TS(kbest,s)=TS_tmp(bestmodind);      
        % Put the treated pair of modules in the inventory array 
        mod_done(ibest,jbest)=1; 
        rn=length(mod_done(~isnan(mod_done)));
        % Compute time-shifted version of selected temporal module
        newd=TS(kbest,s);
        if newd>=0
            Wishifted=zeros(T,1);% positively time-shifted module
            Wishifted(newd+1:end,:)=Wi(1:T-newd,ibest);
        else
            Wishifted=zeros(T,1);% negatively time-shifted module
            Wishifted(1:T+newd,:)=Wi(-newd+1:end,ibest); 
        end           
        % Subtract the selected pair of spatial-temporal modules 
        % from the input matrix Mbs
        Mbs=Mbs-Wishifted*A((s-1)*P+ibest,jbest)*Wb(jbest,:); 
    end
end

end %#EoF updateTS
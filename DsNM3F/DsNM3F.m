function [Wi,Acal,A,Wb,VAF,E2,ENMF,ELDAw,ELDAb,ELDA]=DsNM3F(Mb,P,N,S,G,gamma,delta,algo,iniguess)
%------------------------------------------------------------------------
% Discriminative sample-based Non-negative Matrix Tri-Factorization
%------------------------------------------------------------------------

%--- Description
% Input params:
%   - Mb = input matrix composed: vertical concatenation of the recorded
%          data across multiple episodes (size T*S x M)
%   - P  = number of temporal (or row) modules to be extracted
%   - N  = number of spatial (or column) modules to be extracted
%   - S  = number of samples (i.e. episodes, trials...)
%   - G  = group labels (integers)
%   - gamma = within-class tuning parameter
%   - delta = between-class tuning parameter
%   - algo = chosen method ('mult', 'als' or 'nlp')
%   - iniguess = to provide an initial guess for Wi, A and Wb (optional)
% Output:
%   - Wi   = P temporal modules
%   - Acal = activation coefficients (cell array)
%   - A    = 2D matrix concatenating all activation coefficients
%   - Wb   = N spatial modules
%   - VAF  = variance accounted for
%   - E2   = total cost evolution (across iterations)
%   - ENMF = NMF-based cost
%   - ELDAw= within-class LDA-based cost
%   - ELDAb= between-class LDA-based cost
%   - ELDA = total LDA-based cost

%--- GNU GPL Licence
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version. See <http://www.gnu.org/licenses/>.

% If you use this software for your research, please refer to the paper entitled 
% "Task-discriminative space-by-time factorization of muscle activity", 
%  by I. Delis, S. Panzeri, T. Pozzo and B. Berret

%--- Contact the authors
%  Bastien Berret (bastien.berret@u-psud.fr; http://hebergement.u-psud.fr/berret/) 
%  Ioannis Delis (ioannis.delis@glasgow.ac.uk)

%------------------------------------------------------------------------
% GENERAL OPTIONS
%------------------------------------------------------------------------
VERBOSE=1; % Print some information
DISPLAYITER=1; % Display or not some information at each iteration
FTOL=1e-8; % Tolerance on the cost function
MAXITER=5000; % Maximum iteration before stopping is enforced
NSTOPS=5; % Number of steps for the stopping criterion

%-------------------------------------------------------------------------
% DATASET CHARACTERISTIC
%-------------------------------------------------------------------------
%---- Get the number of time frames
if rem(size(Mb,1),S)==0
    T=size(Mb,1)/S; % number of temporal dimensions (e.g. time frames)
else
   error(['Please check your input matrix... It should be [T*S, M] with ' ...
          'T=number of time steps, S=number of samples and M=number of muscles']);  
end
%---- Get the number of muscles
M=size(Mb,2); % number of spatial dimensions (e.g. EMG channels)
% S is the number of episodes (total number of repetitions/trials)
%---- Get the block transpose of Mb
Mi=blockTranspose(Mb,'r',S);
%---- Get the number of groups
NG=max(G); % maximum element of G is the number of groups    

%---- Constant useful to compute the VAF
SST=0;
Mmean=zeros(T,M);
for s=1:S, Mmean=Mmean+Mi(:,M*(s-1)+1:M*s)/S; end
for s=1:S
 SST=SST+norm(Mi(:,M*(s-1)+1:M*s)-Mmean,'fro')^2;
end

if VERBOSE,
    disp(['Start extracting ' num2str(P) ' temporal modules and ' ...
           num2str(N) ' spatial modules']);
end

%---- Initialization of arrays
if nargin<9,
    iniguess.Wb=rand(N,M);
    iniguess.Wb=iniguess.Wb./repmat(sum(iniguess.Wb,2),[1 size(iniguess.Wb,2)]); % Normalize rows to one   
    iniguess.Wi=rand(T,P);       
    iniguess.Wi=iniguess.Wi./repmat(sum(iniguess.Wi,1),[size(iniguess.Wi,1) 1]); % Normalize columns to one
    iniguess.A=rand(P,N*S);
end

totaltime=tic;
if ~strcmpi(algo,'nlp')
    
    Wb=iniguess.Wb;
    Wi=iniguess.Wi;         
    A=iniguess.A;   
    
    %--- Error container
    err=NaN(1,MAXITER);
    errNMF=NaN(1,MAXITER); errLDAw=NaN(1,MAXITER); 
    errLDAb=NaN(1,MAXITER); errLDA=NaN(1,MAXITER);

    %--- Initialize some variables
    count=0; it=0;
    telapsed=zeros(1,MAXITER);

    %---- Main iterative loop of the algorithm  
    %     N.B.: we decompose the data as Mi=Wi*A*Wb (for each sample)
    while (count<NSTOPS && it<MAXITER) 
        tic; 
        it=it+1; 

        %-------------------------------------------------------------------------
        % Alternating Least-Square method
        %-------------------------------------------------------------------------
        if strcmpi(algo,'als')
            
            % 1st STEP ----------------------------------------------------------            
            %--- UPDATE Wb 
            %--- We approximate Mb=Mi^{\prime}=Cb^{\prime}*Wb=Cbb*Wb   
               
            Cb=Wi*A; 
            Cbb=blockTranspose(Cb,'c',S);
            
            if numel(Mi)<=1e4,
                % Method 1 : quadratic programming (ensures the decrease of error but more memory consuming!)
                ce=zeros(N,N*M);
                for i=1:N
                    ce(i,i:N:M*N)=1;
                end
                H=kron(eye(M),Cbb); 
                options=optimset('Display','off','Algorithm','interior-point-convex');
                X = quadprog((H'*H),-reshape(Mb,[T*M*S 1])'*H,[],[],ce,ones(N,1),zeros(N*M,1),zeros(M*N,1)+Inf,[],options);
                Wb=reshape(X,[N M]); 
            else
                % Method 2 : constrained least-square (but no non-negativity
                % imposed on the solution, similar to matlab nnmf...)
                Ab_=[ kron(eye(M),2*(Cbb'*Cbb)) kron(ones(M,1),eye(N)) ; kron(ones(1,M),eye(N)) zeros(N)];
                Bb_=[ reshape(2*Cbb'*Mb,[M*N 1]) ; ones(N,1)];  
                Xb=Ab_\Bb_ ; % Solves for Ab_*Xb=Bb_
                % True solution is : Wb=reshape(Xb(1:end-N),[N M]);
                % But we have to keep non-negativity and sum to one
                Wb=max(eps,reshape(Xb(1:end-N),[N M])); % Keep non-negative
                Wb=Wb./repmat(sum(Wb,2),[1 size(Wb,2)]); % Normalize rows to one         
            end            

            % 2nd STEP ----------------------------------------------------------            
            %--- UPDATE Wi
            %--- We approximate Mi=Wi*Cii 

            Atil=blockTranspose(A,'c',S);
            Ci=Atil*Wb;
            Cii=blockTranspose(Ci,'r',S);

             if numel(Mi)<=1e4,
                % Method 1 : quadratic programming (ensures the decrease of error but more memory consuming!)
                ce=zeros(P,T*P);
                for i=1:P
                    ce(i,(i-1)*T+1:i*T)=1;
                end
                H=kron(Cii',eye(T)); 
                options=optimset('Display','off','Algorithm','interior-point-convex');
                X = quadprog((H'*H),-H'*reshape(Mi,[T*M*S 1]),[],[],ce,ones(P,1),zeros(T*P,1),zeros(T*P,1)+Inf,[],options);
                Wi=reshape(X,[T P]); 
             else
                % Method 2 : constrained least-square (but no non-negativity
                % imposed on the solution, similar to matlab nnmf...)
                Ai_=[ kron(2*(Cii*Cii'),eye(T)) kron(eye(P),ones(T,1)) ; kron(eye(P),ones(1,T)) zeros(P)];
                Bi_=[ reshape(2*Mi*Cii',[T*P 1]) ; ones(P,1)];   
                % Solve for A_*X=B_
                Xi=Ai_\Bi_ ; % Least-square solution
                % True solution is : Wi=reshape(Xi(1:end-P),[T P]);
                % But we have to keep non-negativity and sum to one
                Wi=max(eps,reshape(Xi(1:end-P),[T P])); % Keep non-negative
                Wi=Wi./repmat(sum(Wi,1),[size(Wi,1) 1]); % Normalize columns to one
             end
      
        %-------------------------------------------------------------------------
        % Multiplicative rules method
        %-------------------------------------------------------------------------    
        else
              
            % 1st STEP ----------------------------------------------------------            
            %--- UPDATE Wb 
            %--- We approximate Mb=Mi^{\prime}=Cb^{\prime}*Wb=Cbb*Wb   
            Cb=Wi*A; 
            Cbb=blockTranspose(Cb,'c',S);
            numer=Cbb'*Mb;
            denom=(Cbb'*Cbb)*Wb;
            Wb= Wb.*(numer+eps(numer))./(denom+eps(numer));    
            Wb=Wb./repmat(sum(Wb,2),[1 size(Wb,2)]); % Normalize rows to one   
            
            % 2nd STEP ----------------------------------------------------------            
            %--- UPDATE Wi
            %--- We approximate Mi=Wi*Cii      

            Atil=blockTranspose(A,'c',S);
            Ci=Atil*Wb;
            Cii=blockTranspose(Ci,'r',S);
            numer=Mi*Cii';
            denom=Wi*(Cii*Cii');
            Wi= Wi.*(numer+eps(numer))./(denom+eps(numer));     
            Wi=Wi./repmat(sum(Wi,1),[size(Wi,1) 1]); % Normalize columns to one
        end
        
        % 3rd STEP --------(common to 'mult' and 'als')--------------------------            
        %--- UPDATE A for all samples s=1..S

        % Compute AmeanAll and Amean
        [AmeanAll,Amean]=computeAmeans(A,P,N,G);

        % Update As          
        for s=1:S
            gs=G(s); % group of trial s
            NGi=sum(G==gs); % number of elements in that particular group
            Ameansum=zeros(P,N);
            for n=1:N
             Ameansum(:,n)=sum(Amean(:,n:N:end),2); 
            end
            addnumer=(delta/NGi+gamma)*Amean(:,(gs-1)*N+1:gs*N) ...
                     +delta*NG/S*AmeanAll;
            adddenom=delta/NGi*AmeanAll+gamma*A(:,N*(s-1)+1:N*s) ...
                     +delta/S*Ameansum;
            denom=(Wi'*Wi)*Atil(P*(s-1)+1:P*s,:)*(Wb*Wb') + adddenom ;
            numer=Wi'*Mi(:,M*(s-1)+1:M*s)*Wb' + addnumer ;
            As_=A(:,N*(s-1)+1:N*s); % store current As
            A(:,N*(s-1)+1:N*s)=A(:,N*(s-1)+1:N*s).*((numer+eps(numer))./(denom+eps(numer)));  
            % Update Amean AND AmeanAll because we modified As 
            AmeanAll=AmeanAll-1/S*(As_-A(:,N*(s-1)+1:N*s));
            Amean(:,(gs-1)*N+1:gs*N)=Amean(:,(gs-1)*N+1:gs*N)-1/NGi*(As_-A(:,N*(s-1)+1:N*s));
        end

        %---- Clean the Output, by reordering or rescaling etc.
        [errNMFit,errLDAwit,errLDAbit] = computeError(Mi,Wi,A,Wb,gamma,delta,P,N,S,G);

        errNMF(it)=errNMFit;
        errLDAw(it)=errLDAwit;
        errLDAb(it)=errLDAbit;
        errLDA(it)=errLDAwit+errLDAbit;
        err(it)=errNMF(it)+errLDA(it);

        telapsed(it)=toc;
        if DISPLAYITER && VERBOSE,
            disp(['iter #' num2str(it) ' | Error=' num2str( err(it)) ' | Time (s)=' num2str(telapsed(it))]);
        end

        %---- Implement convergence criterion
        if it>2
            if (abs(err(it)-err(it-1))<FTOL)
                count=count+1;
                if DISPLAYITER && VERBOSE,
                    disp(['stop counter #' num2str(count)]);
                end
            elseif err(it)>err(it-1)
                warning('DsNM3F:errorIncreased','Error increased during this iteration...');
                it=it-1;
                break  
            elseif err(it)<0
                it=it-1;
                break    
            else
                count=0;
            end
        end 

    end

    E2=err(1:it);
    ENMF=errNMF(1:it);
    ELDAw=errLDAw(1:it);
    ELDAb=errLDAb(1:it);
    ELDA=errLDA(1:it);
    VAF=1-ENMF(end)/SST;

    % Order the elements
    [A,Wi,Wb] = orderOutput(A,Wi,Wb,P,N,S);

    %---- Build the full Acal cell array of activations
    Acal=cell(1,S);
    for s=1:S
        Acal{s}=A(:,N*(s-1)+1:s*N);
    end

%-------------------------------------------------------------------------
% Nonlinear programming method
%-------------------------------------------------------------------------    
else
        
    % Build initial guess
    z0=[reshape(iniguess.Wi,[T*P, 1]) ;reshape(iniguess.A,[N*P*S, 1]); reshape(iniguess.Wb,[M*N, 1])];  

    p.gamma=gamma; p.delta=delta;
    p.Mb=Mb; p.S=S; p.N=N; p.P=P; p.MSC=M;
    p.T=T; p.G=G; p.NG=NG;

    %------- FMINCON solution
    if numel(z0)>=200
       options = optimset('DerivativeCheck','off','Display','iter','TolFun',FTOL,'MaxFunEval',1e3*MAXITER,'MaxIter',MAXITER,...
                       'GradObj','on','Algorithm','interior-point','Hessian',{'lbfgs',5},'TolCon',FTOL,'GradConstr','on');         
    else
       options = optimset('DerivativeCheck','off','Display','iter','TolFun',FTOL,'MaxFunEval',1e3*MAXITER,'MaxIter',MAXITER,...
                       'GradObj','on','Algorithm','interior-point','GradConstr','on'); % or if preferred: 'Algorithm','active-set'     
    end

    res = fmincon(@(z) nlpCostFun(z,p),z0,[],[],[],[], ...
                       zeros(T*P+N*P*S+M*N,1),[ones(T*P,1); N*P*max(max(Mb))*ones(N*P*S,1); ones(M*N,1)], ...
                       @(z) nlpConstFun(z,p),options);

    % Reshape the output
    Wi=reshape(res(1:T*P),[T P]);
    Wb=reshape(res(end-M*N+1:end),[N M]);

    Avectres=zeros(P*N,S);
    for s=1:S
       As=reshape(res(T*P+P*N*(s-1)+1:T*P+P*N*s),[P,N]); 
       Avectres(:,s)=reshape(As,[P*N,1]);
    end
    A=reshape(Avectres,[P N*S]);

    % Order the elements
    [A,Wi,Wb] = orderOutput(A,Wi,Wb,P,N,S);

    %---- Build the full Acal cell array of activations
    Acal=cell(1,S);
    for s=1:S
        Acal{s}=A(:,N*(s-1)+1:s*N);
    end

    [errNMF,errLDAw,errLDAb] = computeError(Mi,Wi,A,Wb,gamma,delta,P,N,S,G);

    ELDA=errLDAw+errLDAb;
    ENMF=errNMF;
    ELDAw=errLDAw;
    ELDAb=errLDAb;
    E2=ENMF+ELDA;
    VAF=1-errNMF/SST;
        
end 

toctotaltime=toc(totaltime);
if VERBOSE,
 disp(['Finished! VAF=' num2str(VAF) ' | ELDAw=' num2str(ELDAw(end)) ' | ELDAb=' num2str(ELDAb(end)) ' | ENMF=' num2str(ENMF(end)) ' | Etotal=' num2str(E2(end)) ]);
 disp(['Total time elapsed ' num2str(toctotaltime) ' seconds']); 
end

end %#EoF DsNM3F

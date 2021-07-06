function [Wi,Acal,Wb,TS,VAF,E2]=sNM3F(Mb,P,N,S)
%------------------------------------------------------------------------
% Sample-based non-negative matrix tri-factorization with (negative or
% positive) time-shifts
%------------------------------------------------------------------------

%--- Description
% Input:
%   - Mb = input matrix defined as the vertical concatenation of the 
%          recorded data across multiple episodes (size T*S x M)
%   - P  = number of temporal (or row) modules to be extracted
%   - N  = number of spatial (or column) modules to be extracted
%   - S  = number of samples (i.e. trials, episodes...)
% Output:
%   - Wi   = P temporal modules (sample-independent)
%   - Wb   = N spatial modules (sample-independent)
%   - Acal = activation coefficients (sample-dependent)
%   - TS   = matrix of time-shifts (to deal with temporal delays)
%   - VAF  = variance accounted for
%   - E2   = total reconstruction error squared (across iterations)

%--- GNU GPL Licence
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version. See <http://www.gnu.org/licenses/>.

% If you use this software for your research, please refer to the paper entitled 
% "A unifying model of concurrent spatial and temporal modularity in muscle activity", 
%  by I. Delis, S. Panzeri, T. Pozzo and B. Berret

%--- Contact the authors
%  Bastien Berret (bastien.berret@u-psud.fr - https://sites.google.com/site/bberret) 
%  Ioannis Delis (ioannis.delis@glasgow.ac.uk)

%-------------------------------------------------------------------------
% VERIFICATION OF DATASET COMPATIBILITY
%-------------------------------------------------------------------------
if rem(size(Mb,1),S)==0
    T=size(Mb,1)/S; % number of temporal dimensions (e.g. time frames)
else
   error(['Please check your input matrix... It should be [T*S, M] with ' ...
          'T=number of time steps, S=number of samples and M=number of muscles']); 
end
M=size(Mb,2); % number of spatial dimensions (e.g. muscles)
% S is the number of samples (total number of repetitions/trials)


%-------------------------------------------------------------------------
% sNM3F ALGORITHM PARAMETERS - USER CAN EDIT HERE
%-------------------------------------------------------------------------
MAXITER=500; % Maximum iteration before stopping is enforced
ERRTOL=1e-5; % Tolerance on the recontruction error changes
NSTOPS=3; % Number of steps for the stopping criterion
CLEANOUTPUT=1; % Normalize the output at each iteration - Do not change the error, simply clean and order the output
DISPLAYITER=0; % Display or not some information at each iteration
ACTIVETS=1; % Active the time-shifts update rule after a number of iterations / Set to '0' to de-activate time-shifts
MAXTS=round(T/2)-1; % Maximum time-shift allowed
MINTS=0; % e.g. '0' or '-MAXTS' if negative time-shifts are allowed (N.B.: -MINTS+MAXTS<T must hold)
TSfun='updateTS'; % function for updating the TS (e.g.: use 'updateTS' or 'updateTSfast')

%-------------------------------------------------------------------------
% sNM3F ALGORITHM (possibly with time-shifts)
%-------------------------------------------------------------------------
disp(['Start extracting ' num2str(P) ' temporal modules and ' ...
       num2str(N) ' spatial modules']);

%--- Error container
err=NaN(1,MAXITER);

%--- Initialize the procedure with a random guess
Wb=rand(N,M);
Wi=rand(T,P);         
A=rand(P*S,N);
if ACTIVETS>1
  TS=round(MINTS/2+rand(P*N,S)*MAXTS/2);
else
  TS=zeros(P*N,S);  
end

%--- Build the Ahat matrix (only required if CLEANOUTPUT is false)
if ~CLEANOUTPUT, Ahat=buildAhat(A,P,N,S);  end %#ok<NASGU>

%--- Initialize some variables
count=0; it=0;
telapsed=zeros(1,MAXITER);

%--- Create the matrices to deal with Time-Shifts
disp('Building matrices to deal with time shifts...');
if -MINTS+MAXTS>=T
   error('Maximum time-shifts allowed must satisfy -MINTS+MAXTS<T'); 
end
D=buildDs(T,max(abs([MAXTS MINTS])));

%--- Create Theta and Thetaij once for all if Time-Shifts are not activated
if ACTIVETS==0,
    Theta=buildTheta(TS,P,N,S,D);
    Thetaij=buildThetaij(TS,T,P,N,S);
    Q=kron(speye(S),repmat(Wi,[1 N])); %#ok<NASGU>
end

%---- Main iterations of the algorithm
%     N.B.: the global factorization is Mb = Q(Theta,Wi)*Ahat*Wb
while (count<NSTOPS && it<MAXITER) % stopping criterion: if the error does not change for 5 iterations or 200 iterations are exceeded
   tic;
   it=it+1; % counter for the main iterations
  
   %---- Preserve a clean output (may improve robustness & convergence)      
   if CLEANOUTPUT,
     [A,Wi,Wb,TS]=cleanOutput(A,Wi,Wb,TS,P,N,S,T);
     Ahat=buildAhat(A,P,N,S); % Matrix A has changed so Ahat should do so
   end
   
   % 1st STEP -------------------------------------------------------
   %---- Update the Time-Shifts if activated
   if ACTIVETS>0,
    if it>=ACTIVETS
      % Consider using a mex file if 'updateTS.m' is too slow
      try    
        TS=eval([TSfun '_mex(TS,Mb,A,Wi,Wb,T,M,P,N,S,MAXTS,MINTS)']); 
        if it==1, disp('Time-shifts are updated with a mex file'); end
      catch %#ok<CTCH>
        TS=eval([TSfun '(TS,Mb,A,Wi,Wb,T,M,P,N,S,MAXTS,MINTS)']);   
        if it==1, disp(['If too slow, consider using a mex file for ' TSfun '.m']); end
      end
    end 

    % Update Theta, Thetaij and Q (since they all depend on 'TS')
    Theta=buildTheta(TS,P,N,S,D);
    Thetaij=buildThetaij(TS,T,P,N,S);
    Q=buildQ(Theta,Wi,T,P,N,S);
   else
    if CLEANOUTPUT,
      % Q must still be updated here (because of the call to cleanOutput)
      Q=kron(speye(S),repmat(Wi,[1 N])); % [TS x PNS]
    end
   end % End If 'ACTIVETS>0'

   % 2nd STEP -------------------------------------------------------
   %---- UPDATE Wb 
   QA=Q*Ahat; QAt=QA';
   numer=QAt*Mb;
   denom=(QAt*QA)*Wb;
   Wb= Wb.*numer./(denom+eps(numer));
   
   % 3rd STEP -------------------------------------------------------
   %---- UPDATE Wi
   AWb=Ahat*Wb; % [PNS x M]
   QAWbt=(Q*AWb)';
   Mbt=Mb';
   for i=1:T
       for j=1:P
        ThetaijAWb=Thetaij{i,j}*AWb;
        numer=trace(Mbt*ThetaijAWb);
        denom=trace(QAWbt*ThetaijAWb);   
        Wi(i,j)=Wi(i,j)*numer/(denom+eps(numer));   
       end
   end
   
   % Update Q since Wi has changed
   if ACTIVETS>0,
     Q=buildQ(Theta,Wi,T,P,N,S); % since Wi has just been updated
   else
     Q=kron(speye(S),repmat(Wi,[1 N])); % [TS x PNS]  
   end

   % 4th STEP -------------------------------------------------------
   %---- UPDATE A, given that Mb=Q*A*Wb
   for s=1:S
       Qs=Q((s-1)*T+1:s*T,(s-1)*P*N+1:s*P*N);
       Qst=Qs';
       Ahats=Ahat((s-1)*P*N+1:s*P*N,:);
       Mbs=Mb((s-1)*T+1:s*T,:);
       numer=Qst*Mbs*Wb';
       denom=(Qst*(Qs*Ahats*Wb)*Wb');
       for i=1:P
           for j=1:N
              k=i+(j-1)*P;
              A((s-1)*P+i,j)=A((s-1)*P+i,j)*numer(k,j)/...
                                           (denom(k,j)+eps(numer(k,j))); 
           end
       end
   end
   
   Ahat=buildAhat(A,P,N,S); % Update Ahat since A has been updated
   
   % 5th STEP -------------------------------------------------------
   %---- Mean reconstruction error (across all samples)
   err(it)=norm(Mb-Q*Ahat*Wb,'fro')^2;
 
   % Computation time for the current iteration
   telapsed(it)=toc;
   
   if DISPLAYITER,
   disp(['iter #' num2str(it) ' | Error=' num2str( err(it)) ...
            ' | Time (s) ' num2str(telapsed(it))]);
   end

   %---- Implement stopping criterion
   if it>2
    if (abs(err(it)-err(it-1))<ERRTOL)
        count=count+1;
        if DISPLAYITER,
            disp(['stop event #' num2str(count)]);
        end
     elseif err(it)>err(it-1)
        count=count+1;
        if DISPLAYITER,
            disp(['stop event #' num2str(count)]);
        end
     else
        count=0;
     end
   end
end %- End While


%---- Return the final Error and Compute the VAF
E2=err(1:it);
VAF=getVAF(Mb,E2(end),S);

%---- Clean the output solution (normalize, re-order etc.)
if CLEANOUTPUT,
    [A,Wi,Wb,TS]=cleanOutput(A,Wi,Wb,TS,P,N,S,T);
end

%---- Build the full Acal cell array of activations
Acal=cell(1,S);
for s=1:S
    Acal{s}=A(P*(s-1)+1:s*P,:);
end

disp(['Finished! VAF is ' num2str(VAF) ' | Err is ' num2str(E2(end))]);
disp(['Total time elapsed ' num2str(sum(telapsed)) ' seconds']);
fprintf('\n');

end % #EoF sNM3F
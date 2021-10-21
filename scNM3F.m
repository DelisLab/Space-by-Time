function [Wi,Acal,Wb,VAF,E2]=scNM3F(Mb,P,N,S)
%------------------------------------------------------------------------
% Convex sample-based non-negative matrix tri-factorization
%------------------------------------------------------------------------

%--- Description
% Input params:
%   - Mb = input matrix composed: vertical concatenation of the recorded
%          data across multiple episodes (size T*S x M)
%   - P = number of temporal (or row) modules to be extracted
%   - N = number of spatial (or column) modules to be extracted
%   - S = number of episodes (i.e. samples, trials...)
% Output:
%   - Wi = P non-negative temporal modules
%   - Wb = N non-negative spatial modules
%   - Acal = activation coefficients (can be negative)
%   - VAF = variance accounted for
%   - E2 = total reconstruction error

%--- GNU GPL Licence
% if you use it for your research, please refer to the corresponding paper.
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%--- Contact the authors
%  Ioannis Delis (delisyannis@gmail.com)
%  Arno Onken (aonken@gmail.com)
%  Bastien Berret (b.berret@gmail.com)




%-------------------------------------------------------------------------
% scNM3F_basic ALGORITHM PARAMETERS - YOU CAN EDIT HERE
%-------------------------------------------------------------------------

MAXITER=1000; % Maximum iteration before stopping is enforced
ERRTOL=1e-6; % Tolerance on the recontruction error changes
CLEANOUTPUT=1; % Normalize the output at each iteration
DISPLAYITER=1; % Display or not some information at each iteration
NSTOPS=10; % Number of steps for the stopping criterion

%-------------------------------------------------------------------------
% DATASET CHARACTERISTIC
%-------------------------------------------------------------------------
if rem(size(Mb,1),S)==0
    T=size(Mb,1)/S; % number of temporal dimensions (e.g. time frames)
else
   error('Please check your input matrix...'); 
end
M=size(Mb,2); % number of spatial dimensions (e.g. EMG channels)
% S is the number of episodes (total number of repetitions/trials)


%---- Get the block transpose of Mb
Mi=blocktranspose(Mb,'r',S);


%-------------------------------------------------------------------------
% BASIC sNM3F ALGORITHM
%-------------------------------------------------------------------------

disp(['Start extracting ' num2str(P) ' temporal modules and ' num2str(N) ' spatial modules']);

%---- Initialization of arrays - YOU CAN EDIT THE INITIAL GUESS
Wb=rand(N,M);
Wi=rand(T,P);         
A=rand(P,N*S);  
 
%--- Error container
err=NaN(1,MAXITER);

count=0; it=0;
telapsed=zeros(1,MAXITER);

%---- Main iterative loop of the algorithm     
while (count<NSTOPS && it<MAXITER) 
    tic; 
    it=it+1;
    %-- we decompose the data as Mi=Wi*A*Wb (for each sample)
         
    %---- Clean the Output, by reordering or rescaling etc.
    % May improve the robustness and convergence
    if CLEANOUTPUT,
     % Norm rows and colums to one
     [A,Wi,Wb] = NormalizeOutput(A,Wi,Wb,N,P,S);
%      if ORTHOGONALIZE, 
      % Ensure we get matrices of the order of identity for Wi'*Wi and Wb*Wb'
      [A,Wi,Wb] = rescaleOutput(A,Wi,Wb);  
%      end
     % Then, order the elements
     [A,Wi,Wb] = OrderOutput(A,Wi,Wb,P,N,S);
    end
    
    % 1st STEP ----------------------------------------------------------            
    %--- UPDATE Wb using cluster-NMF objective

    B=Mb'*Mb*Wb';
    Bp=(abs(B)+B)./2;
    Bn=(abs(B)-B)./2;
    C=(Mb*Wb')'*Mb*Wb';
    Cp=(abs(C)+C)./2;
    Cn=(abs(C)-C)./2;
    Wb=Wb.*sqrt((Bp+Wb'*Cn)./(Bn+Wb'*Cp+eps))';

    % 2nd STEP ----------------------------------------------------------            
    %--- UPDATE Wi using cluter-NMF objective 

    B=Mi*(Wi'*Mi)';
    Bp=(abs(B)+B)./2;
    Bn=(abs(B)-B)./2;
    C=Wi'*Mi*(Wi'*Mi)';
    Cp=(abs(C)+C)./2;
    Cn=(abs(C)-C)./2;
    Wi=Wi.*sqrt((Bp+Wi*Cn)./(Bn+Wi*Cp+eps));

    % 3rd STEP ----------------------------------------------------------            
    %--- UPDATE A for all samples s=1..S and compute the error err(it)
    err(it)=0;
    Wi_inv = pinv(Wi);
    Wb_inv = pinv(Wb);
    for tr=1:S
        % Apply pseudoinverse of Wi and Wb to compute A
    	A(:,N*(tr-1)+1:N*tr)= Wi_inv*Mi(:,M*(tr-1)+1:M*tr)*Wb_inv;
        % compute single-trial approximation error
    	err(it)=err(it)+sum(sum((Mi(:,M*(tr-1)+1:M*tr)-Wi*A(:,N*(tr-1)+1:N*tr)*Wb).^2));
    end
     
    telapsed(it)=toc;
    if DISPLAYITER,
    	disp(['iter #' num2str(it) ' | Error=' num2str( err(it)) ' | Time (s)=' num2str(telapsed(it))]);
    else
        if mod(it,50)==0, fprintf('.'); end
	end
                 
%     ---- Implement convergence criterion
    if it>2
    	if (abs(err(it)-err(it-1))<ERRTOL)
            count=count+1;
            if DISPLAYITER,
                disp(['stop counter #' num2str(count)]);
            end
        elseif err(it)>err(it-1)
            warning('sNM3F:errorIncreased','Error increased during this iteration...');
            break
        else
            count=0;
        end
    end 
                
end

fprintf('\n');

SST=0;
for tr=1:S
 SST=SST+ sum(sum((Mi(:,M*(tr-1)+1:M*tr)-mean(mean(Mi(:,M*(tr-1)+1:M*tr)))).^2));
end

VAF=1-err(it)/SST;
E2=err(it);

%---- Clean the Output, by reordering or rescaling etc.
if CLEANOUTPUT,
 % Norm rows and colums to one
 [A,Wi,Wb] = NormalizeOutput(A,Wi,Wb,N,P,S);
 % Then, order the elements
 [A,Wi,Wb] = OrderOutput(A,Wi,Wb,P,N,S);
end

Acal=zeros(P,N,S);
for tr=1:S
 Acal(:,:,tr)=A(:,N*(tr-1)+1:tr*N);
end

disp(['Finished! VAF is ' num2str(VAF) ' | Err is ' num2str(E2)]);
disp(['Total time elapsed ' num2str(sum(telapsed)) ' seconds']);     

end 



%-------------------------------------------------------------------
% Blocktranspose function
%-------------------------------------------------------------------

function Mat_out=blocktranspose(Mat_in,type,S)

[r,c]=size(Mat_in);
if strcmpi(type,'r') % in rows
    Mat=[];
    M=r/S;
    for s=1:S
            Mat=horzcat(Mat,Mat_in(M*(s-1)+1:M*s,:)); 
    end
elseif strcmpi(type,'c') % in columns
    Mat=[];
    M=c/S;
    for s=1:S
            Mat=vertcat(Mat,Mat_in(:,M*(s-1)+1:M*s)); 
    end
else
    error('Unknown....')
end
Mat_out=Mat;

end

%-------------------------------------------------------------------
% CleanOutput functions
%-------------------------------------------------------------------

function [A,Wi,Wb] = NormalizeOutput(A,Wi,Wb,N,P,S)

% Row-wise normalization
sP=zeros(1,P);
for i=1:P
   sP(i)=norm(Wi(:,i));
   Wi(:,i)=Wi(:,i)./sP(i);
end
% Column-wise normalization
sN=zeros(1,N);
for j=1:N
   sN(j)=norm(Wb(j,:));
   Wb(j,:)=Wb(j,:)./sN(j);
end
% Element-wise normalization of A to make the error unchanged
for tr=1:S
    for i=1:P
       for j=1:N  
        A(i,N*(tr-1)+j)=A(i,N*(tr-1)+j).*(sP(i)*sN(j));
       end
    end
end

end


function [A,Wi,Wb] = RescaleOutput(A,Wi,Wb)

% Make sure that Wi'*Wi and Wb*Wb' scales to identity
sPm=max(max(Wi'*Wi));
Wi=Wi./sqrt(sPm);
sNm=max(max(Wb*Wb'));
Wb=Wb./sqrt(sNm);
A=A.*sqrt(sPm)*sqrt(sNm);

end


function [Anew,Wi,Wb] = OrderOutput(A,Wi,Wb,P,N,S)

Anew=zeros(size(A));

% Order elements by the occurence of the maximal element
% in columns for Wi
for i=1:P
    [ignore,iP(i)]=max(Wi(:,i));
end
[ignore,idxP] = sort(iP,'ascend');

% in rows for Wb
for j=1:N
    [ignore,iN(j)]=max(Wb(j,:));
end
[ignore,idxN] = sort(iN,'ascend');

Wi=Wi(:,idxP);
Wb=Wb(idxN,:);

for tr=1:S
    for j=1:N
        Anew(:,N*(tr-1)+j)=A(:,N*(tr-1)+idxN(j));
    end
end
Anew(:,:)=Anew(idxP,:);

end

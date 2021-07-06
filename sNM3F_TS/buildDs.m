function D=buildDs(T,MAXTS)
% Build the time-shift matrices, and store them into a structure
% Time-shifts can vary between 0 to +MAXTS
% N.B.: for a negative time-shift, we shall use the transpose of Ds{d+1}

D=cell(1,2*MAXTS+1);
D{1}=speye(T); % Identity matrix when no delay

if MAXTS>=1 && MAXTS<T
    for d=1:MAXTS
       subDiag=zeros(1,T); subDiag(d+1)=1;
       D{d+1}=sparse(toeplitz(subDiag,zeros(1,T))); % returns a subdiag matrix
    end
end

end %#EoF buildDs
clear all
clc

% Load some example data
load input.mat

Mi=blockTranspose(Mb,'r',S);

% Number of modules extracted
P=2;
N=2;

gamma=1.3; % within-class parameter
delta=0.8; % between-class parameter

REP=1; % number of repetitions: the larger, the better

st.temporalModules=[]; st.spatialModules=[];
st.combinators=[]; st.VAF=[]; st.ERR=[]; 
st.ENMF=[]; st.ELDA=[]; st.ELDAw=[]; st.ELDAb=[];
res=repmat(st,1,REP); ERR=nan(1,REP);

for r=1:REP       
    
    % Main DsNM3F extraction
    [Wi,Acal,A,Wb,VAF,E2,ENMF,ELDAw,ELDAb,ELDA]=DsNM3F(Mb,P,N,S,G,gamma,delta,'als'); 

    % Build a result structure
    res(r).temporalModules=Wi;
    res(r).spatialModules=Wb;
    res(r).combinators=Acal;
    res(r).A=A; res(r).VAF=VAF;

    res(r).ENMF=ENMF; res(r).ELDAw=ELDAw; res(r).gamma=gamma;
    res(r).ELDAb=ELDAb; res(r).ELDA=ELDA; res(r).delta=delta;  

    res(r).ERR=sqrt(E2(end)/numel(Mb));
    res(r).E2=E2;
    ERR(r)=E2(end);
    
    Avectres=zeros(P*N,S);
    for s=1:S
     Avectres(:,s)=reshape(A(:,(s-1)*N+1:s*N),[P*N,1]);
    end
    res(r).Avect=Avectres;
    
end

% Return the best run (with respect to the total cost function)
[~,indr]=min(ERR);
RES=res(indr);

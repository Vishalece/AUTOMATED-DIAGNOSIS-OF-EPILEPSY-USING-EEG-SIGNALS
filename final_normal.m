norm1 = table2array(readtable('O.csv'));
norm2 = table2array(readtable('Z.csv'));
REN1=0;
ShEn1 = 0;
TE1 = 0;
ApEn1 = 0;

REN2=0;
ShEn2= 0;
TE2 = 0;
ApEn2 = 0;

r1=[];r2=[];r=[];
s1=[];s2=[];s=[];
t1=[];t2=[];t=[];
a1=[];a2=[];a=[];
total=[];
for i= 1 : 100

    data1 = norm1(:,i);
    N = length(data1);
    Y1 = fft(data1); % DFT of signal
    p1 = abs(Y1/N).^2; % power spectral density
    % Calculate the Renyi entropy using the formula.
    w1= - log2(sum(p1.^2));
    REN1 = REN1 +w1;
    r1=[r1 w1];

    y1 = -sum(p1.*log2(p1))
    ShEn1 = ShEn1 + y1;
    s1=[s1 y1];

    step1=data1.^2;
    x1 = 1 - sum(step1);
    TE1 = TE1 + x1;
    t1=[t1 x1];

    En1 = approximateEntropy(data1,[],2);
    ApEn1 = En1 + ApEn1;
    a1=[a1 En1];

    data2 = norm2(:,i);
    Y2 = fft(data2); % DFT of signal
    p2 = abs(Y2/N).^2; % power spectral density
    % Calculate the Renyi entropy using the formula.
    w2 = - log2(sum(p2.^2));
    REN2 = REN2 +w2;
    r2=[r2 w2];

    y2 = -sum(p2.*log2(p2))
    ShEn2 = ShEn2 + y2;
    s2=[s2 y2];


    step2 = data2.^2;
    x2= 1 - sum(step2);
    TE2 = TE2 + x2;
    t2=[t2 x2];
    
    
    En2 = approximateEntropy(data2,[],3);
    ApEn2 = En2 + ApEn2;
    a2=[a2 En2];
end
r =[r1 r2];
s=[s1 s2];
t=[t1 t2];
a=[a1 a2];
total=[r
    s
    t
    a];
xlswrite("final_normal.xlsx",total);
RENmean = (REN1 + REN2)/200
ShEnmean = (ShEn1 + ShEn2)/200
TEmean = (TE1 + TE2)/200
ApEnmean = (ApEn1 + ApEn2)/200
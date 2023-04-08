norm1 = table2array(readtable('S.csv'));

REN1=0;
ShEn1 = 0;
TE1 = 0;
ApEn1 = 0;



r1=[];
s1=[];
t1=[];
a1=[];
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

    En1 = approximateEntropy(data1,[],3);
    ApEn1 = En1 + ApEn1;
    a1=[a1 En1];

   
end

total=[r1
    s1
    t1
    a1];
xlswrite("final_ictal.xlsx",total);
RENmean = (REN1 )/100
ShEnmean = (ShEn1)/100
TEmean = (TE1)/100
ApEnmean = (ApEn1)/100
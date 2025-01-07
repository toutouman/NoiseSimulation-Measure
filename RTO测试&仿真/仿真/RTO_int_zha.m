%%
NoiseOmega = 0;NoiseOmega1 = 0;NoiseOmega2 = 0.0;
NoiseFreq = 100*2*pi;NoiseFreq1 = 7*2*pi;NoiseFreq2 = 39*2*pi;
interval = 400e-6;
integ = 500e-9;
Nall = 1e6;

h1 = figure();ax1 = axes(h1);hold on;
h2 = figure();ax2 = axes(h2);hold on;
eventX = nan(1,Nall);
eventY = nan(1,Nall);
XX = nan(1,Nall);
YY = nan(1,Nall);
tlist = interval:interval:interval*Nall;
PPList = nan(1,Nall);
parfor ii = 1:length(tlist)
    t = tlist(ii);
    
    PP = NoiseOmega*cos(NoiseFreq*t)+NoiseOmega1*cos(NoiseFreq1*t)+NoiseOmega2*cos(NoiseFreq2*t);
%     PP = PP+random('uniform',-1,1)*100;
    PPList(ii) = PP;
    StateVector = exp(1j*PP);
    YTemp = (imag(StateVector)+1)/2;
    rdY = random('uniform',0,1);
    YY(ii) = YTemp;
    if rdY<YTemp
        eventY(ii) = 1;
    else
        eventY(ii) = 0;
    end
end


windows = 100;
deltaT = interval*windows;




PYY = nan(1,length(eventY)/windows);
parfor ii =1:length(PYY)
    PYY(ii) = mean(eventY((ii-1)*windows+1:ii*windows));
end
PY = PYY;
Yline = PY*2-1;
Xline = sqrt(1-(Yline).^2);

angleXY = (angle(Xline+1j*(Yline)))/integ/1;


% PYY = nan(1,length(PPList)/windows);
% parfor ii =1:length(PYY)
%     PYY(ii) = mean(PPList((ii-1)*windows+1:ii*windows));
% end
% angleXY = PYY/integ/1;

Pallfft = fft(angleXY);
PSD = (2*pi)^2*(abs(Pallfft)).^2*deltaT^2/length(angleXY)/deltaT;
PSD(1) = PSD(1)/2;
freq1 = linspace(0,1/deltaT,length(angleXY));
freq1 = freq1(2:int32((length(freq1)-1)/2));
PSD = PSD(2:int32((length(PSD)-1)/2));

plot(ax1,freq1,PSD)
% title([TitleName,' PSD'])
% xlim([-xl,xl]);
xlabel(ax1,'Frequency (Hz)')
% ylim([0,2]);
ylabel(ax1,'Amplitude (a.u.)')
set(ax1,'Yscale','log') 
set(ax1,'Xscale','log')

NMax = ceil(log10(freq1(end)));
NMin = floor(log10(freq1(1)));
NList = NMin:1:NMax;
freqAvg = nan(1,int32(9*(NMax-NMin)));
PSDAvg = nan(1,int32(9*(NMax-NMin)));
for ii=1:length(NList)
    kk = NList(ii);
    for jj = 1:9
        index = find((freq1>jj*10^kk).*(freq1<(jj+1)*10^kk));
        freqAvg((ii-1)*9+jj) = mean(freq1(index));
        PSDAvg((ii-1)*9+jj) = mean(PSD(index));
    end

end

freqAvg(isnan(freqAvg))=[];
PSDAvg(isnan(PSDAvg))=[];
plot(ax2,freqAvg,PSDAvg,'marker','o')
xlabel(ax2,'Frequency (Hz)')
% ylim([0,2]);
ylabel(ax2,'Amplitude (a.u.)')
set(ax2,'Yscale','log') 
set(ax2,'Xscale','log')

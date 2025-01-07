NoiseOmega = 0;NoiseOmega1 = 0;NoiseOmega2 = 1.0;
NoiseFreq = 100*2*pi;NoiseFreq1 = 7*2*pi;NoiseFreq2 = 39*2*pi;
interval = 200e-6;
Nall = 1e5;
eventX = [];
eventY = [];
XX = [];
YY = [];
tlist = interval:interval:interval*Nall;
PPList = [];
for ii = 1:length(tlist)
    t = tlist(ii);
    
%     t = t+random('uniform',-1,1)*5e-6;
    PP = NoiseOmega*cos(NoiseFreq*t)+NoiseOmega1*cos(NoiseFreq1*t)+NoiseOmega2*cos(NoiseFreq2*t);
%     PP = random('norm',0,10);
%     PP = pi/2*exp(-(t^2)/(1e-1^2));
%     PP = 0;

    PPList(end+1) = PP;
    StateVector = exp(1j*PP);
    YTemp = (imag(StateVector)+1)/2;
    rdY = random('uniform',0,1);
    YY(end+1) = YTemp;
    if rdY<YTemp
        eventY(end+1) = 1;
    else
        eventY(end+1) = 0;
    end
end


window = 1%ceil(2*pi/NoiseFreq/interval/10);
% window = 10;
time = interval*window;
PY = [];

NSel = length(YY)-window;
for ii = 1:NSel
    dataSelY = eventY(ii:ii+window-1);
    PYY = mean(dataSelY);
    PY(end+1) = PYY;   
end
Yline = PY*2-1;
%Xline = sqrt(1-(Yline).^2);
angleXY = asin(Yline);%(angle(Xline+1j*(Yline)));
angleXY=PPList(1:NSel);


Pallfft = fft(angleXY);
Pallfft = Pallfft/NSel*2;%要除以N/2，以保证幅度
% Pallfft(1) = Pallfft(1);
Pallfft1 = fftshift(Pallfft);
freq1 = linspace(-1/interval/2,1/interval/2,NSel);
figure();
plot(freq1,abs(Pallfft1))
title(['window=',num2str(window)])
% xlim([-50,50]);
xlabel('Frequency (Hz)')
% ylim([0,0.1]);
ylabel('Amplitude (a.u.)')
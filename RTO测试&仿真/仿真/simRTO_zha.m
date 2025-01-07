NoiseOmega = 1;NoiseOmega1 = 1;NoiseOmega2 = 0;
NoiseFreq = 100*2*pi;NoiseFreq1 = 2000*2*pi;NoiseFreq2 = 39*2*pi;
interval = 200e-6;
Nall = 1e5;
eventX = [];
eventY = [];
XX = [];
YY = [];
tlist = interval:interval:interval*Nall;
for ii = 1:length(tlist)

    t = tlist(ii);
    StateVector = exp(1j*(NoiseOmega*cos(NoiseFreq*t)));
    if mod(ii,2)==1
        XTemp = (real(StateVector)+1)/2;
        rdX = random('uniform',0,1);
        XX(end+1) = XTemp;
        if rdX<XTemp
            eventX(end+1) = 1;
        else
            eventX(end+1) = 0;
        end
    else
        YTemp = (imag(StateVector)+1)/2;
        rdY = random('uniform',0,1);
        YY(end+1) = YTemp;
        if rdY<YTemp
            eventY(end+1) = 1;
        else
            eventY(end+1) = 0;
        end
    end
end


window = ceil(2*pi/NoiseFreq/interval/2/50);
window = 1;
time = interval*2*window;
PX = [];
PY = [];

NSel = length(XX)-window;
for ii = 1:NSel
    dataSelX = eventX(ii:ii+window-1);
    dataSelY = eventY(ii:ii+window-1);
    PXX = mean(dataSelX);
    PYY = mean(dataSelY);
    PX(end+1) = PXX;
    PY(end+1) = PYY;
%     PY(end+1) = (sqrt(1-(PXX*2-1)^2)+1)/2;
    
end
angleXY = (angle(PX*2-1+1j*(PY*2-1)));
% figure();
% plot(angleXY)
% title(num2str(window))

Pallfft = fft(angleXY);
Pallfft = Pallfft/NSel*2;%要除以N/2，以保证幅度
Pallfft(1) = Pallfft(1)/2;
Pallfft1 = fftshift(Pallfft);
freq1 = linspace(-1/interval/2/2,1/interval/2/2,NSel);
figure();
plot(freq1,abs(Pallfft1))
title(['window=',num2str(window)])
% xlim([-500,500]);
xlabel('Frequency (Hz)')
ylim([0,1.1]);ylabel('Amplitude (a.u.)')
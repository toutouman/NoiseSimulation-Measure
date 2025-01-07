

%%
% clear all
q='Q12';
omega_0=5.068e9;
num_round=1000;
quanti_noise=6.39e+10/(2*pi)^4;
window=100;
interval = 400e-6;
Nall=1e4;
% Q_list=["Q01","Q02","Q03","Q04","Q05","Q06"];
Q_list=["Q07","Q08","Q09","Q10","Q11","Q12"];
q_order=find(Q_list==q);

load_name=['RTO_',q,'_KC5210_100M'];
[PSD,freq1,PSDAvg,freqAvg,angleXY]=Cal_RTO_PSD(load_name,num_round,window,quanti_noise);
figure();
loglog(freq1,PSD)
figure();
X_angle=1:1:length(angleXY);
plot(X_angle,angleXY);

Ec_list=[256e6,256e6,256e6,256e6,256e6,256e6];
% Omega_maxR=[5.128e9,5.124e9,5.059e9,5.100e9,5.092e9,5.188e9];
Omega_maxR=[5.197e9,5.102e9,5.233e9,5.1095e9,5.081e9,5.143e9];
omega_max=Omega_maxR(q_order);
[k,delta]=Find_flux_k(omega_max,Ec_list(q_order),omega_0);
% k=1;
% delta=0;
freqAvg(isnan(freqAvg))=[];
PSDAvg(isnan(PSDAvg))=[];
h2 = figure();ax2 = axes(h2);hold on;
plot(ax2,freqAvg,PSDAvg./k^2,'marker','o')
xlabel(ax2,'Frequency (Hz)')
% ylim([0,2]);
ylabel(ax2,'Amplitude (a.u.)')
set(ax2,'Yscale','log') 
set(ax2,'Xscale','log')
legend([q,' ',num2str(round(delta/1e6)),' MHz'])

%%
q='Q05';
detune='200M';

Q_list=["Q04","Q07","Q08","Q10"];
detune_list=["100M","200M"];
interval = 400e-6;

Nall = 1e4;
num=999;
event_list=[];
tic
for i=1:num 
    Event_data=load(['RTO_',q,'_STa035_200M (',num2str(i),').mat'], 'basis');
    event_i=str2num(Event_data.basis);
    event_list=[event_list;event_i];
end
toc
% event_list=load(['RTO_event_list (',num2str(100),').mat'], 'basis').basis
window = 100%ceil(2*pi/NoiseFreq/interval/10);
% window = 10;
time = interval*window;
PY = [];

NSel = round(length(event_list)/window);
for ii = 1:NSel
    dataSelY = event_list((ii-1)*window+1:ii*window);
    PYY = mean(dataSelY);
    PY(end+1) = PYY;   
end
Yline = PY*2-1;
Xline = sqrt(1-(Yline).^2);
angleXY = (angle(Xline+1j*(Yline)))/500e-9/2/pi;


Pallfft = fft(angleXY);
% Pallfft = Pallfft/NSel*2;
% Pallfft(1) = Pallfft(1)/2;
Pallfft1 = fftshift(Pallfft);

deltaT = interval*window;

PSD=(abs(Pallfft1)).^2*deltaT^2/length(angleXY)/deltaT;
PSD(1)=PSD(1)/2;
freq1 = linspace(-1/interval/window/2,1/interval/window/2,NSel);
figure();
loglog(-freq1(1:round(NSel/2)),PSD(1:round(NSel/2)))
title(['window=',num2str(window)])
figure();
loglog(freq1(round(NSel/2)+1:round(NSel)),PSD(round(NSel/2)+1:round(NSel)))
title(['window=',num2str(window)])
% xlim([-50,50]);
% xlabel('Frequency (Hz)')
% % ylim([0,0.1]);
% ylabel('Amplitude (a.u.)')
%%
%计算采样误差
% quanti_noise= RTO_Quanti_Noise(1e6,window,interval);
quanti_noise=6.39e+10/(2*pi)^4
% quanti_noise=0
%计算平均噪声功率
NMax = ceil(log10(freq1(end)));
NMin = floor(log10(freq1(round(NSel/2+1))));
NList = NMin:1:NMax;
freqAvg = nan(1,int32(9*(NMax-NMin)));
PSDAvg = nan(1,int32(9*(NMax-NMin)));
for ii=1:length(NList)
    kk = NList(ii);
    for jj = 1:9
        index = find((freq1>jj*10^kk).*(freq1<(jj+1)*10^kk));
        freqAvg((ii-1)*9+jj) = mean(freq1(index));
        PSDAvg((ii-1)*9+jj) = mean(PSD(index))-quanti_noise;
    end

end

%计算磁通斜率
Ec_list=[256e6,250e6,252e6,252e6];
%Sta034
Zpa2f01_list=[[-0.14876,-5364.07,4.4403e9];[-0.0658, -4548.9, 4.2796e+9];
        [-0.0525,-4118.9,4.3704e9];[-0.1783,-9197,4.45527e9]];
Omega0_list_high=[4.436e9,4.264e9,4.3573e9,4.448e9];
Omega0_list_low=[4.176e9,4.155e9,4.23e9,4.269e9];
%Sta035
% Zpa2f01_list=[[-0.115733,-4864.67,4.4504e9];[-0.06023, -4523.55, 4.2786e+9];
%         [-0.167149,-12218.95.9,4.01345e9];[-0.1783,-9197.57,4.4552e9]];
% Omega0_list_high=[4.118e9,4.132e9,4.2286e9,4.26919e9];
% Omega0_list_low=[4.017e9,4.0328e9,4.3573e9,4.4483e9];
Omega0_list=[Omega0_list_high;Omega0_list_low];
detune_order=find(detune_list==detune);
q_order=find(Q_list==q);
[k,delta]=Find_flux_k(Zpa2f01_list(q_order,:),Ec_list(q_order),Omega0_list(detune_order,q_order))
% k=1
%Plot
h2 = figure();ax2 = axes(h2);hold on;
freqAvg(isnan(freqAvg))=[];
PSDAvg(isnan(PSDAvg))=[];
plot(ax2,freqAvg,PSDAvg/k^2,'marker','o')
xlabel(ax2,'Frequency (Hz)')
% ylim([0,2]);
ylabel(ax2,'Amplitude (a.u.)')
set(ax2,'Yscale','log') 
set(ax2,'Xscale','log')
legend([q,' ',num2str(round(delta/1e6)),' MHz'])
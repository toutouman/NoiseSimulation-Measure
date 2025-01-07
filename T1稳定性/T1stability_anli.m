
clear all

q='Q04';
% detune='100M';
Q_list=["Q03","Q04"];
% detune_list=["100M","200M"];
interval = 400e-6;
Nall = 1e4;
num=1000;
event_list=[];
tic
for i=1:num 
    Event_data=load(['T1stability_Sta035_',q,' (',num2str(i),').mat'], 'basis');
    event_i=str2num(Event_data.basis);
    event_list=[event_list;event_i];
end
toc
mean_P=mean(event_list);
% event_list=load(['RTO_event_list (',num2str(100),').mat'], 'basis').basis
window = 100%ceil(2*pi/NoiseFreq/interval/10);
% window = 10;
time = interval*window;
P_1 = [];

NSel = round(length(event_list)/window);
for ii = 1:NSel
    dataSelY = event_list((ii-1)*window+1:ii*window);
    P_ii = mean(dataSelY);
    P_1(end+1) = P_ii;   
end
% Yline = PY*2-1;
% Xline = sqrt(1-(Yline).^2);
% angleXY = (angle(Xline+1j*(Yline)))/500e-9;





Pallfft = fft(P_1);
% Pallfft = Pallfft/NSel*2;
% Pallfft(1) = Pallfft(1)/2;

Pallfft1 = fftshift(Pallfft);
deltaT = interval*window;
PSD = (abs(Pallfft)).^2*deltaT^2/length(P1_m)/deltaT;
PSD(1) = PSD(1)/2;
freq1 = linspace(0,1/deltaT,length(P1_m));
freq1 = freq1(2:int32((length(freq1)-1)/2));
PSD = PSD(2:int32((length(PSD)-1)/2));


PSD=(2*pi)^2*(abs(Pallfft1)).^2*deltaT^2/length(P_1)/deltaT;
PSD(1)=PSD(1)/2;
freq1 = linspace(-1/interval/window/2,1/interval/window/2,NSel);
figure();
loglog(-freq1(1:round(NSel/2)),PSD(1:round(NSel/2)))
% semilogy(freq1,PSD)
title(['window=',num2str(window)])
% xlim([-50,50]);
% xlabel('Frequency (Hz)')
% % ylim([0,0.1]);
% ylabel('Amplitude (a.u.)')
figure()
plot(event_list)
%计算采样误差
% quanti_noise= RTO_Quanti_Noise(1e6,window,interval);
% quanti_noise=6.39e+10
quanti_noise=0
%计算平均噪声功率
NMax = ceil(log10(freq1(end)));
NMin = floor(log10(freq1(round(NSel/2+1))));
NList = NMin:1:NMax;
freqAvg = nan(1,int32(9*(NMax-NMin)));
PSDAvg = nan(1,int32(9*(NMax-NMin)));
for ii=1:length(NList)
    kk = NList(ii);
    for jj = 1:9
        index = find((-freq1>jj*10^kk).*(-freq1<(jj+1)*10^kk));
        freqAvg((ii-1)*9+jj) = mean(-freq1(index));
        PSDAvg((ii-1)*9+jj) = mean(PSD(index))-quanti_noise;
    end

end

%计算磁通斜率
Ec_list=[256e6,257e6];
Zpa2f01_list=[[-0.1638,-7374,4.1377e9];[-0.167149, -12218.9, 4.01345e+9];
        [-0.057894,-7097.9,4.2286e9];[-0.191156,-15005,4.26919e9]];
Omega0_list_high=[4.118e9,4.132e9,4.2286e9,4.26919e9];
Omega0_list_low=[4.017e9,4.0328e9,4.3573e9,4.4483e9];
Omega0_list=[Omega0_list_high;Omega0_list_low];
% detune_order=find(detune_list==detune);
q_order=find(Q_list==q);
% [k,delta]=Find_flux_k(Zpa2f01_list(q_order,:),Ec_list(q_order),Omega0_list(detune_order,q_order))
k=1
%Plot
h2 = figure();ax2 = axes(h2);hold on;
freqAvg(isnan(freqAvg))=[];
PSDAvg(isnan(PSDAvg))=[];
plot(ax2,freqAvg,PSDAvg,'marker','o')
xlabel(ax2,'Frequency (Hz)')
% ylim([0,2]);
ylabel(ax2,'Amplitude (a.u.)')
set(ax2,'Yscale','log') 
set(ax2,'Xscale','log')
legend([q])

% 
% a_0=2.5e-9;
% b_0=0.95;
% c_0=0;
% ex={'x','1'};
% % 拟合1/f噪声
% for i=1:10
%     S_f=fittype('a./(f^0.9)+c','independent','f','coefficients',{'a','c'});
%     [S_fit,ana]=fit(freqAvg(1,2:end-4)',PSDAvg(1,2:end-4)'/k^2,S_f,'StartPoint',[a_0 c_0]);
%     a_0=S_fit.a;
% %     b_0=S_fit.b;
%     c_0=S_fit.c;
% end
% a=S_fit.a;
% % b=S_fit.b;
% c=S_fit.c;
% Fit_Noise=a*1.6./freqAvg.^0.9+0;
% % Fit_Noise=S_fit(freqAvg)
% plot(ax2,freqAvg,Fit_Noise)
% 
%%
clear all
% Load_Q={'Q07','Q08','Q09','Q10','Q11','Q12'};
% Load_Q={'Q01','Q02','Q03','Q04','Q05','Q06'};
Load_Q={'Q01','Q02','Q03','Q04','Q05','Q06','Q07','Q08','Q09','Q10','Q11','Q12'};
num_round=1000;
window=100;
Mean_P=[]
h1 = figure();ax1 = axes(h1);hold on;
h2 = figure();ax2 = axes(h2);hold on;
Quanti_N= 1.0e-04 *[0.9048  0.9009 0.9052 0.8609 0.9019 0.9400 0.9146 0.8826 0.9256 0.8733 0.9067 0.8812]
Quanti_N= 1.0e-04 *[0.8656  0.9102 0.9080 0.8632 0.9375 0.9315 0.8834 0.9083 0.9127 0.8773 0.8863 0.8615]
for i=1:length(Load_Q)
    q=Load_Q{i};
    quanti_noise=Quanti_N(i);
    loadname=['T1stability_Sta035_',q];
    [PSD,freq1,PSDAvg,freqAvg,mean_P]=Cal_T1s_PSD(loadname,num_round,window,quanti_noise);
    Mean_P(end+1)=mean_P
    freqAvg(isnan(freqAvg))=[];
    PSDAvg(isnan(PSDAvg))=[];
    plot(ax1,freq1,PSD);
    plot(ax2,freqAvg,PSDAvg,'marker','o');
end
xlabel(ax2,'Frequency (Hz)');
% ylim([0,2]);
ylabel(ax2,'Amplitude (a.u.)');
legend(ax1,Load_Q);
legend(ax2,Load_Q);
set(ax1,'Yscale','log') 
set(ax1,'Xscale','log')
set(ax2,'Yscale','log') 
set(ax2,'Xscale','log')
Quanti_N=[]
for i=1:length(Load_Q)
    mean_P=Mean_P(i);
    quanti_noise= T1s_Quanti_Noise(5e6,window,mean_P);
    Quanti_N(end+1)=quanti_noise
end



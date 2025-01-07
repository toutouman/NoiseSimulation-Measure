function [PSD,freq1,PSDAvg,freqAvg,mean_P]=Cal_T1s_PSD(load_name,num_round,window,quanti_noise)
    if (nargin<4)
        quanti_noise=0;
    end
    interval = 400e-6;
    % num=1000;
    event_list=[];
    for i=1:num_round 
        Event_data=load([load_name,' (',num2str(i),').mat'], 'basis');
        event_i=str2num(Event_data.basis);
        event_list=[event_list;event_i];
    end
    mean_P=mean(event_list);
    time = interval*window;
    P_1 = [];
    NSel = round(length(event_list)/window);
    for ii = 1:NSel
        dataSelY = event_list((ii-1)*window+1:ii*window);
        P_ii = mean(dataSelY);
        P_1(end+1) = P_ii;   
    end
    deltaT = interval*window;
    Pallfft = fft(P_1);
    % Pallfft = Pallfft/NSel*2;
    % Pallfft(1) = Pallfft(1)/2;
    PSD = (abs(Pallfft)).^2*deltaT^2/length(P_1)/deltaT;
    PSD(1) = PSD(1)/2;
    freq1 = linspace(0,1/deltaT,length(P_1));
    freq1 = freq1(2:int32((length(freq1)-1)/2));
    PSD = PSD(2:int32((length(PSD)-1)/2));
    %     Pallfft1 = fftshift(Pallfft);
%     figure();
%     loglog(freq1,PSD)
    %计算采样误差
%     if is_cutQuasiN
%         quanti_noise=quanti_noise;
%     else quanti_noise=0;
%     end

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
            PSDAvg((ii-1)*9+jj) = mean(PSD(index))-quanti_noise;
        end

    end
%     h2 = figure();ax2 = axes(h2);hold on;
%     freqAvg(isnan(freqAvg))=[];
%     PSDAvg(isnan(PSDAvg))=[];
%     plot(ax2,freqAvg,PSDAvg,'marker','o')
%     xlabel(ax2,'Frequency (Hz)')
%     % ylim([0,2]);
%     ylabel(ax2,'Amplitude (a.u.)')
%     set(ax2,'Yscale','log') 
%     set(ax2,'Xscale','log')
%     legend([q,' ',num2str(round(delta/1e6)),' Hz'])
end

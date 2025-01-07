
function [PSD,freq1,PSDAvg,freqAvg,angleXY1]=Cal_RTO_PSD(load_name,num_round,window,quanti_noise)
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
%     event_list=Del_ReadError(event_list);
    time = interval*window;
    PY = [];
    NSel = floor(length(event_list)/window);
    for ii = 1:NSel
        dataSelY = event_list((ii-1)*window+1:ii*window);
        PYY = mean(dataSelY);
        PY(end+1) = PYY;   
    end
    Yline = PY*2-1;
    Xline = sqrt(1-(Yline).^2);
    angleXY = (angle(Xline+1j*(Yline)))/500e-9/2/pi+4.5e9;
    angleXY1 = (angle(Xline+1j*(Yline)));
    deltaT = interval*window;
    Pallfft = fft(angleXY);
    PSD = (abs(Pallfft)).^2*deltaT^2/length(angleXY)/deltaT;
    PSD(1) = PSD(1)/2;
    freq1 = linspace(0,1/deltaT,length(angleXY));
    freq1 = freq1(2:int32((length(freq1)-1)/2));
    PSD = PSD(2:int32((length(PSD)-1)/2));
    %     Pallfft1 = fftshift(Pallfft);
%     figure();
%     loglog(freq1,PSD)
    

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

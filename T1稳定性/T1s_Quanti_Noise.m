%%


function quanti_noise= T1s_Quanti_Noise(Nall,windows,mean_P)
%     integ = 500e-9;
    interval=400e-6;
    h2 = figure();ax2 = axes(h2);hold on;
    eventP1 = nan(1,Nall);
    P_1 = nan(1,Nall);
    tlist = interval:interval:interval*Nall;
%     PPList = nan(1,Nall);
    parfor ii = 1:length(tlist)
        t = tlist(ii);
        PP = mean_P;
%         PPList(ii) = PP;
        r_1 = random('uniform',0,1);
        P_1(ii) = PP;
        if r_1<PP
            eventP1(ii) = 1;
        else
            eventP1(ii) = 0;
        end
    end

%     windows = 100;
    deltaT = interval*windows;
    P1_m = nan(1,length(eventP1)/windows);
    parfor ii =1:length(P1_m)
        P1_m(ii) = mean(eventP1((ii-1)*windows+1:ii*windows));
    end

    Pallfft = fft(P1_m);
    PSD = (abs(Pallfft)).^2*deltaT^2/length(P1_m)/deltaT;
    PSD(1) = PSD(1)/2;
    freq1 = linspace(0,1/deltaT,length(P1_m));
    freq1 = freq1(2:int32((length(freq1)-1)/2));
    PSD = PSD(2:int32((length(PSD)-1)/2));

%     plot(ax1,freq1,PSD)
    % title([TitleName,' PSD'])
    % xlim([-xl,xl]);
%     xlabel(ax1,'Frequency (Hz)')
    % ylim([0,2]);
%     ylabel(ax1,'Amplitude (a.u.)')
%     set(ax1,'Yscale','log') 
%     set(ax1,'Xscale','log')

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
    xlabel(ax2,'Frequency (Hz)');
    % ylim([0,2]);
    ylabel(ax2,'Amplitude (a.u.)');
    set(ax2,'Yscale','log') ;
    set(ax2,'Xscale','log');
    quanti_noise=mean(PSDAvg(15:end));
end
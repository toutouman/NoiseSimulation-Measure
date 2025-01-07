
clear all
repet=6;

IQ0Gc_list=[];
IQ1Gc_list=[];
P00_list=[];
P11_list=[];
SNR_list=[];
% IQ0_Buffer=[]
tic
for i=1:repet 
    Event_data_0=load(['readoutStability_iq0_',num2str(i),'.mat']);
    Event_data_1=load(['readoutStability_iq1_',num2str(i),'.mat']);
    IQ0Gc_list=[IQ0Gc_list;Event_data_0.iq0Gc];
    IQ1Gc_list=[IQ1Gc_list;Event_data_1.iq1Gc];
    P00_list=[P00_list;Event_data_0.P0];
    P11_list=[P11_list;Event_data_1.P1];
    for j=1:length(Event_data_0.P0)
        SNR_i=[];
        P00_list_i=[];
        P11_list_i=[];
        for k =1:3
            IQ0_list_ij=Event_data_0.iq0Buffer{k+(j-1)*3,2};
            IQ1_list_ij=Event_data_1.iq1Buffer{k+(j-1)*3,2};
            C_0=Event_data_0.iq0Gc(j,k);C_1=Event_data_1.iq1Gc(j,k);
            [P00_j,P11_j]=Readout_Fidelity(IQ0_list_ij,IQ1_list_ij,C_0,C_1);
            
%             [FWHM0_j,IQ0_C_j]=RO_GaussFit(IQ0_list_ij);
%             [FWHM1_j,IQ1_C_j]=RO_GaussFit(IQ1_list_ij);
%             SNR_j=abs(IQ0_C_j-IQ1_C_j)/((FWHM0_j+FWHM1_j)/2);
%             SNR_i=[SNR_i,SNR_j];
%             P00_list_i=[P00_list_i,P00_j];
%             P11_list_i=[P11_list_i,P11_j];
        end
%         P00_list=[P00_list;P00_list_i];
%         P11_list=[P11_list;P11_list_i];
%         SNR_list=[SNR_list;SNR_i];
    end
end
toc
num=length(P00_list(:,1));
time_list=linspace(0,4,num);
P00_Avg=mean(P00_list);
P00_Std=std(P00_list);
P11_Avg=mean(P11_list);
P11_Std=std(P11_list);
% SNR_Avg=mean(SNR_list);
% SNR_Std=std(SNR_list);

Q_list=["Q01","Q03","Q05"];
h1 = figure();ax1 = axes(h1);hold on;
h2 = figure();ax2 = axes(h2);hold on;
legend_list1=[];
legend_list2=[];
for i=1:3
    plot(ax1,time_list,P00_list(:,i));
    legend_list1=[legend_list1;[Q_list(i)+" P00="+num2str(P00_Avg(i))+"\pm"+num2str(P00_Std(i))]];
    plot(ax1,time_list,P11_list(:,i));
    legend_list1=[legend_list1;[Q_list(i)+" P11="+num2str(P11_Avg(i))+"\pm"+num2str(P11_Std(i))]];
%     plot(ax2,time_list,SNR_list(:,i));
%     legend_list2=[legend_list2;[Q_list(i)+" SNR="+num2str(SNR_Avg(i))+"\pm"+num2str(SNR_Std(i))]];
end
legend(ax1,legend_list1);
xlabel(ax1,'Time (Hour)');
ylabel(ax1,'Fidelity');
legend(ax2,legend_list2);
xlabel(ax2,'Time (Hour)');
ylabel(ax2,'SNR');

IQ0Gc_Avg=mean(IQ0Gc_list);
IQ0Gc_Std_R=std(real(IQ0Gc_list));
IQ0Gc_Std_I=std(imag(IQ0Gc_list));
IQ0Gc_Std=IQ0Gc_Std_R+1j*IQ0Gc_Std_I;
IQ1Gc_Avg=mean(IQ1Gc_list);
% IQ1Gc_Std=std(IQ1Gc_list);
IQ1Gc_Std_R=std(real(IQ1Gc_list));
IQ1Gc_Std_I=std(imag(IQ1Gc_list));
IQ1Gc_Std=IQ1Gc_Std_R+1j*IQ1Gc_Std_I;
figure();
legend_list=[];
for i=1:3
    I0=real(IQ0Gc_list);
    Q0=imag(IQ0Gc_list);
    plot(I0(:,i),Q0(:,i));
    hold on;
    legend_list=[legend_list;[Q_list(i)+"|0\rangle IQ Center (1e6): "+num2str(round(IQ0Gc_Avg(i)/1e6,2))+" \pm("+num2str(round(IQ0Gc_Std(i)/1e6,2))+"), RF: "+num2str(round(abs(IQ0Gc_Std(i))/(abs(IQ0Gc_Avg(i)))*1e2,2))+"%"]];
    I1=real(IQ1Gc_list);
    Q1=imag(IQ1Gc_list);
    plot(I1(:,i),Q1(:,i));
    hold on;
    legend_list=[legend_list;[Q_list(i)+"|1\rangle IQ Center (1e6): "+num2str(round(IQ1Gc_Avg(i)/1e6,2))+" \pm("+num2str(round(IQ1Gc_Std(i)/1e6,2))+"), RF: "+num2str(round(abs(IQ1Gc_Std(i))/(abs(IQ1Gc_Avg(i)))*1e2,2))+"%"]];
end
% plot(real(IQ0_Buffer(1,:)),imag(IQ0_Buffer(1,:)));
legend(legend_list);
xlabel('I');
ylabel('Q');
%%
repet=6;

IQ0Gc_list=[];
IQ1Gc_list=[];
P00_list=[];
P11_list=[];
% IQ0_Buffer=[]
IQ_data_0=load(['readoutStability_iq0_',num2str(1),'.mat']);
IQ_data_1=load(['readoutStability_iq1_',num2str(1),'.mat']);
IQ0Gc_list=IQ_data_0.iq0Gc;
IQ1Gc_list=IQ_data_1.iq1Gc;
IQ0_Buffer=[];
IQ1_Buffer=[];
for i=1:3
    IQ0_Buffer_i=[];
    IQ1_Buffer_i=[];
    n=20;
    for j=1:n
        IQ0_Buffer_i=[IQ0_Buffer_i,IQ_data_0.iq0Buffer{i+(j-1)*3,2}];
        IQ1_Buffer_i=[IQ1_Buffer_i,IQ_data_1.iq1Buffer{i+(j-1)*3,2}];
    end
    IQ0_Buffer=[IQ0_Buffer;IQ0_Buffer_i];
    IQ1_Buffer=[IQ1_Buffer;IQ1_Buffer_i];
end
IQ0C_P=mean(IQ0Gc_list(1:n,:));
IQ1C_P=mean(IQ1Gc_list(1:n,:));
Delta_r=IQ0C_P-IQ1C_P;
Delta_c=(IQ0C_P+IQ1C_P)/2;
Theta_r=atan(imag(Delta_r)./real(Delta_r));

dots_num=length(IQ1_Buffer(1,:));
% IQ0C_new=IQ0C_P
% IQ1C_new=IQ1C_P
Theta_m=repmat(transpose(Theta_r),1,dots_num);
% IQ0_Buffer_new=IQ0_Buffer;
% IQ1_Buffer_new=IQ1_Buffer;
IQ0C_trans_x=real(IQ0C_P).*cos(Theta_r)+imag(IQ0C_P).*sin(Theta_r);
IQ0C_trans_y=-real(IQ0C_P).*sin(Theta_r)+imag(IQ0C_P).*cos(Theta_r);
IQ1C_trans_x=real(IQ1C_P).*cos(Theta_r)+imag(IQ1C_P).*sin(Theta_r);
IQ1C_trans_y=-real(IQ1C_P).*sin(Theta_r)+imag(IQ1C_P).*cos(Theta_r);
IQ0_trans_x=real(IQ0_Buffer).*cos(Theta_m)+imag(IQ0_Buffer).*sin(Theta_m);
IQ0_trans_y=-real(IQ0_Buffer).*sin(Theta_m)+imag(IQ0_Buffer).*cos(Theta_m);
IQ1_trans_x=real(IQ1_Buffer).*cos(Theta_m)+imag(IQ1_Buffer).*sin(Theta_m);
IQ1_trans_y=-real(IQ1_Buffer).*sin(Theta_m)+imag(IQ1_Buffer).*cos(Theta_m);

% sz=0.5;
% for ii=1:3
%     figure()
%     scatter(real(IQ0_Buffer(ii,:)),imag(IQ0_Buffer(ii,:)),sz,[0,0,1]);
%     hold on
%     scatter(real(IQ1_Buffer(ii,:)),imag(IQ1_Buffer(ii,:)),sz,[1,0,0]);
%     hold on
%     scatter(real(IQ0C_P(ii)),imag(IQ0C_P(ii)),50,'y','X');
%     hold on
%     scatter(real(IQ1C_P(ii)),imag(IQ1C_P(ii)),50,'w','X');
%     hold on
%     scatter(real(Delta_c(ii)),imag(Delta_c(ii)),50,'w','X');
% end

sz=0.5;
for ii=1:3
    figure()
    scatter(real(IQ0_Buffer(ii,:)),imag(IQ0_Buffer(ii,:)),sz,[0,0,1]);
    hold on
    scatter(IQ1_trans_x(ii,:),IQ1_trans_y(ii,:),sz,[1,0,0]);
    hold on
    scatter(real(IQ0C_P(ii)),imag(IQ0C_P(ii)),50,'y','X');
    hold on
    scatter(real(B),imag(B),50,'w','X');
end

fit_num=101;
for ii=1:3
    IQ0_all_ii=IQ0_trans_x(ii,:);
    IQ1_all_ii=IQ1_trans_x(ii,:);
    IQ0_1D_i=[0];
    IQ1_1D_i=[0];
    fit0_range=linspace(min(IQ0_all_ii),max(IQ0_all_ii),fit_num);
    fit1_range=linspace(min(IQ1_all_ii),max(IQ1_all_ii),fit_num);
    for jj=1:fit_num-1
        index0_jj=find(IQ0_all_ii>fit0_range(jj) & IQ0_all_ii<=fit0_range(jj+1));
        num0_jj=length(index0_jj);
        numx_jj=length(x_index_jj);
        index1_jj=find(IQ1_all_ii>fit1_range(jj) & IQ1_all_ii<=fit1_range(jj+1));
        num1_jj=length(index1_jj);
        IQ1_1D_i=[IQ1_1D_i,num1_jj];
    end
    F_Gausse=fittype('A*exp(-(x-x_0).^2/(2*C.^2))','independent','x','coefficients',{'A','x_0','C'});
    Fitted_Func0=fit(fit0_range',IQ0_1D_i',F_Gausse,'startpoint',[4000 1e7 1e7]);
    Fitted_Func1=fit(fit1_range',IQ1_1D_i',F_Gausse,'startpoint',[4000 1e7 1e7]);
    figure()
    bar(fit0_range,IQ0_1D_i,'b','edgecolor','none')
    hold on;
    bar(fit1_range,IQ1_1D_i,'r','edgecolor','none')
    hold on;
    plot(fit0_range,Fitted_Func0(fit0_range),'k','Linewidth',2)
    hold on;
    plot(fit1_range,Fitted_Func1(fit1_range),'m','Linewidth',2)
    legend(Q_list(ii)+' |0\rangle',Q_list(ii)+' |1\rangle',['|0\rangle FWHM: ',num2str(round(2.355*Fitted_Func0.C/1e6,2)),'e6'],['|1\rangle FWHM: ',num2str(round(2.355*Fitted_Func1.C/1e6,2)),'e6'])
end


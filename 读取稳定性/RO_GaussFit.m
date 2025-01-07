function [FWHM,IQ_C]=RO_GaussFit(IQ_list)
    fit_num=121;
    IQ_list_x=real(IQ_list);
    IQ_list_y=imag(IQ_list);
    fit_range_x=linspace(min(IQ_list_x),max(IQ_list_x),fit_num);
    fit_range_y=linspace(min(IQ_list_y),max(IQ_list_y),fit_num);
    d_x=abs(fit_range_x(2)-fit_range_x(1));
    d_y=abs(fit_range_y(2)-fit_range_y(1));
    IQ_count_x=[0];
    IQ_count_y=[0];
    for ii=1:fit_num-1
        x_index_ii=find(IQ_list_x>fit_range_x(ii) & IQ_list_x<=fit_range_x(ii+1));
        y_index_ii=find(IQ_list_y>fit_range_y(ii) & IQ_list_y<=fit_range_y(ii+1));
        numx_ii=length(x_index_ii);
        numy_ii=length(y_index_ii);
        IQ_count_x=[IQ_count_x,numx_ii];
        IQ_count_y=[IQ_count_y,numy_ii];
    end
    fit_range_x=fit_range_x-d_x;
    fit_range_y=fit_range_y-d_y;
    F_Gausse=fittype('A*exp(-(x-s_0).^2/(2*C.^2))','independent','x','coefficients',{'A','s_0','C'});
    [Ax_int,sx_int]=max(IQ_count_x);
    Fitted_Funcx=fit(fit_range_x',IQ_count_x',F_Gausse,'startpoint',[Ax_int fit_range_x(sx_int) fit_num*d_x/4]);
    [Ay_int,sy_int]=max(IQ_count_y);
    Fitted_Funcy=fit(fit_range_y',IQ_count_y',F_Gausse,'startpoint',[Ay_int fit_range_y(sy_int) fit_num*d_y/4]);
    IQ_C=Fitted_Funcx.s_0+1j*Fitted_Funcy.s_0;
    FWHM=2.355*(Fitted_Funcx.C+Fitted_Funcy.C)/2;
%     figure()
%     bar(fit_range_x,IQ_count_x,'b','edgecolor','none')
%     hold on
%     plot(fit_range_x,Fitted_Funcx(fit_range_x),'k','Linewidth',2)
%     figure()
%     bar(fit_range_y,IQ_count_y,'r','edgecolor','none')
%     hold on
%     plot(fit_range_y,Fitted_Funcy(fit_range_y),'m','Linewidth',2)
end
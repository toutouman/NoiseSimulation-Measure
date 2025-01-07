function [F_00,F_11]=Readout_Fidelity(iq0_list,iq1_list,C_0,C_1)
    D_x=real(C_0-C_1);
    D_y=imag(C_0-C_1);
    k_c=D_y./D_x;
    k_d=-1./k_c;
    b_d=0.5*imag(C_0+C_1)-k_d*0.5*real(C_0+C_1);
    %分界线为 y=k_d*x+b_d
    C0_y=imag(C_0);C0_x=real(C_0);
    C1_y=imag(C_1);C1_x=real(C_1);
    IQ0_y=imag(iq0_list);IQ0_x=real(iq0_list);
    IQ1_y=imag(iq1_list);IQ1_x=real(iq1_list);
    if C0_y>k_d*C0_x+b_d
        index_00=find(IQ0_y-IQ0_x*k_d>b_d);
        F_00=length(index_00)./length(IQ0_y);
        index_11=find(IQ1_y-IQ1_x*k_d<b_d);
        F_11=length(index_11)./length(IQ1_y);
    else
        index_00=find(IQ0_y-IQ0_x*k_d<b_d);
        F_00=length(index_00)./length(IQ0_y);
        index_11=find(IQ1_y-IQ1_x*k_d>b_d);
        F_11=length(index_11)./length(IQ1_y);
    end
%     figure()
%     sz=0.5;
%     hold on;
%     scatter(IQ0_x,IQ0_y,sz,[0,0,1]);
%     hold on
%     scatter(IQ1_x,IQ1_y,sz,[1,0,0]);
%     hold on;
%     plot([C0_x,C1_x],[C0_x*k_d+b_d,C1_x*k_d+b_d],'k','Linewidth',2);
%     hold on;
%     plot([C0_x,C1_x],[C0_y,C1_y],'g--','Linewidth',2)
        
end
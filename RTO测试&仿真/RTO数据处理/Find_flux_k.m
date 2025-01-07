function [k,delta]=Find_flux_k(omega_max,E_c,omega_0,Zpa2f01)
    if (nargin>3)
        a=Zpa2f01(1);
        b=Zpa2f01(2);
        c=Zpa2f01(3);
        amp_max=-b/2/a;
        omega_max=a*amp_max^2+b*amp_max+c;
    end
    delta=omega_max-omega_0;
    %amp_0=sqrt((omega_0-c)/a+(b/2/a)**2)-b/2/a;
    %k_bias=2*a*amp_0+b;
    e=1.6021766208e-19;
    h=6.626070154e-34;
%     E_c=e^2/2/C/h;
    E_j=(omega_max+E_c)^2/8/E_c;
    Phi_0=h/2/e;
    Phi_ex=acos((omega_0+E_c)^2/(8*E_c*E_j))*Phi_0/pi;
    %Phi_ex=Phi_0*(delta+sqrt(8*E_c*E_j))**2/(8*E_c*E_j)/pi;
    k=-0.5*8*E_c*E_j*pi/Phi_0*sin(pi*Phi_ex/Phi_0)/sqrt(8*E_c*E_j*cos(pi*Phi_ex/Phi_0))*Phi_0;
end
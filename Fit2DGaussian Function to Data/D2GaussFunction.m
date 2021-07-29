function F = D2GaussFunction(x,xdata)
%% x = [Amp, x0, wx, y0, wy, fi]
xdatarot(:,:,1)= xdata(:,:,1)*cos(x(6)) - xdata(:,:,2)*sin(x(6));
xdatarot(:,:,2)= xdata(:,:,1)*sin(x(6)) + xdata(:,:,2)*cos(x(6));
x0rot = x(2)*cos(x(6)) - x(4)*sin(x(6));
y0rot = x(2)*sin(x(6)) + x(4)*cos(x(6));

cova=[x(3)^2,x(3)*x(5);x(5)*x(6),x(5)^2];%(covariance)
nor_cova=norm(reshape(cova,[1,4]));
%F is 2-d gaussian value, size=nxn->pixels
F = x(1)/(2*pi*sqrt(nor_cova))*exp( -((xdatarot(:,:,1)-x0rot).^2/(2*x(3)^2) + (xdatarot(:,:,2)-y0rot).^2/(2*x(5)^2) ));
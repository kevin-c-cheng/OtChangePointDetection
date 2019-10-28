addpath('../Code')
load('person010011_out.mat')
load('TwoSampConvFilter.mat')
filter2=filter2-0.1666;
filter2=filter2/sum(filter2);

L(810)=1;
L(2062)=1;

trace =conv(wStat2Samp,filter2,'same');
[val, cp] = findpeaks(trace);
cpOut=zeros(1,length(L));
cpOut(cp)=1;

figure(4);
clf;
plot(Y(500:end-500+1,:), 'Color', [0.7,0.7,0.7])
yyaxis right
plot(L(500:end-500)*35, 'k--', 'linewidth', 2)
hold on
plot(conv(wStat2Samp,filter2,'same'), 'linewidth', 2.5)
plot(wStat2Samp, 'm-', 'linewidth', 0.25)
plot(cpOut*35, '--', 'linewidth', 2)
xlim([0,10751])
ylim([0,30])

h = findobj(gca,'Type','line');
legend([h(5),h(4), h(3), h(1), h(2)], 'Accel Data', 'GT Change Points', 'W2T Statistic', 'W2T Change Points', 'Unfiltered W2T')

load('SimulatedDistances_2Samp.mat');

cMap = loadColors();
figure(1); clf;
subplot(2,2,1)
plot(wassEmbed(1:200,1), wassEmbed(1:200,2), '.', 'MarkerFace', cMap(1,:))
hold on
plot(wassEmbed(201:400,1), wassEmbed(201:400,2), '.', 'MarkerFace', cMap(2,:))
plot(wassEmbed(401:600,1), wassEmbed(401:600,2), '.', 'MarkerFace', cMap(3,:))
plot(wassEmbed(601:800,1), wassEmbed(601:800,2), '.', 'MarkerFace', cMap(4,:))
title('p=2 Wasserstein')
set(gca,'XTick',[], 'YTick', [])


subplot(2,2,2)
plot(mmdEmbed(1:200,1), mmdEmbed(1:200,2), '.', 'MarkerFace', cMap(1,:))
hold on
plot(mmdEmbed(201:400,1), mmdEmbed(201:400,2), '.', 'MarkerFace', cMap(1,:))
plot(mmdEmbed(401:600,1), mmdEmbed(401:600,2), '.', 'MarkerFace', cMap(2,:))
plot(mmdEmbed(601:800,1), mmdEmbed(601:800,2), '.', 'MarkerFace', cMap(3,:))
title('MMD')
set(gca,'XTick',[], 'YTick', [])


subplot(2,2,3)
plot(w2StatEmbed(1:200,1), w2StatEmbed(1:200,2), '.', 'MarkerFace', cMap(1,:))
hold on
plot(w2StatEmbed(201:400,1), w2StatEmbed(201:400,2), '.', 'MarkerFace', cMap(1,:))
plot(w2StatEmbed(401:600,1), w2StatEmbed(401:600,2), '.', 'MarkerFace', cMap(2,:))
plot(w2StatEmbed(601:800,1), w2StatEmbed(601:800,2), '.', 'MarkerFace', cMap(3,:))
title('W2T')
set(gca,'XTick',[], 'YTick', [])


subplot(2,2,4)
plot(ksEmbed(1:200,1), ksEmbed(1:200,2), '.', 'MarkerFace', cMap(1,:))
hold on
plot(ksEmbed(201:400,1), ksEmbed(201:400,2), '.', 'MarkerFace', cMap(1,:))
plot(ksEmbed(401:600,1), ksEmbed(401:600,2), '.', 'MarkerFace', cMap(2,:))
plot(ksEmbed(601:800,1), ksEmbed(601:800,2), '.', 'MarkerFace', cMap(3,:))
title('K-S')
set(gca,'XTick',[], 'YTick', [])

legend('Normal', 'Laplace', 'Shifted Var', 'Shifted Mean')


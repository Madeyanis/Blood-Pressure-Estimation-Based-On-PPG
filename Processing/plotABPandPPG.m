function [] = plotABPandPPG(ppg, abp)
figure;

subplot(1, 2, 1)
plot(ppg)
subplot(1, 2, 2)
plot(abp)

end
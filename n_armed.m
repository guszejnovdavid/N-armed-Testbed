function n_armed()

%Inputs
EpsVals=[0,0.01,0.1]; %epsilon values to use
PlotColor={'g', 'r', 'k'};
NPlays=1000; %number of plays for a system
NSample=2000; %number of independent systems, for statistics
%Parameters
NArm=10; %number of options (arms)
BiasMean=0.0; %the mean of the distribution from which the biases are drawn
BiasSigma=1.0;%the std. deviation of the distribution from which the biases are drawn
ArmSigma=1.0;%the std. deviation of the random reward from the arms
InitReward=0.0; %Initial reward estimate, high number to encourage exploration

%Statistics
TotalReward=zeros(NPlays,length(EpsVals));
OptimalActionNum=zeros(NPlays,length(EpsVals)); %number of times the optimal action was used

%Epsilon Loop
for i=1:length(EpsVals) %for each epsilon value
    disp(['Calculating for epsilon=' num2str(EpsVals(i)) ]);

%Main loop
    for j=1:NSample %for each system
    %Init
    Bias=random('Normal',BiasMean,BiasSigma,1,NArm); %biases for each arm
    [~,BestArmIndex]=max(Bias); %find optimal index
    Rewards=zeros(1,NArm)+InitReward; %average reward from an "arm"
    NPlayed=zeros(1,NArm); %number of times an "arm" has been played
        for k=1:NPlays %for each play
            %Determine if the choice is stochastic
            if(rand()>EpsVals(i)) %deterministic case
                [~,choice]=max(Rewards); %find best guess
            else %stochastic case
                choice=randi(NArm);
            end
            
            %Play
            NPlayed(choice)=NPlayed(choice)+1;
            currentreward=Bias(choice)+randn()*ArmSigma;%calculate reward
            TotalReward(k,i)=TotalReward(k,i)+currentreward; %update total
            if choice==BestArmIndex
                OptimalActionNum(k,i)=OptimalActionNum(k,i)+1; %was this the optimal choice
            end
            %Update estimates
            if NPlayed(choice)>=2
                Rewards(choice)=(Rewards(choice)*(NPlayed(choice)-1)+currentreward)/NPlayed(choice);
            else
                Rewards(choice)=currentreward;
            end
        end
    end
end

OptimalActionRatio=zeros(NPlays,length(EpsVals));
AverageReward=zeros(NPlays,length(EpsVals));
for i=1:length(EpsVals)
    OptimalActionRatio(:,i)=OptimalActionNum(:,i)./NSample; %ratio of optimal choices
    AverageReward(:,i)=TotalReward(:,i)./NSample; %the average reward per play
end

%%%%%%%
% Plots

titletext=['N armed testbed results for \epsilon-greedy model'];
close all;
figure(1),
set(1,'name',titletext,'units','normalized','position',[0.4,0.25,0.5,0.5])
title(titletext);

subplot(2,1,1);
hold on
for i=1:length(EpsVals)
plot(1:NPlays,AverageReward(:,i),PlotColor{i},'LineWidth',1.5),
end
hold off
ylim([0 max(max(AverageReward))])
xlim([1, NPlays])
xlabel('Plays'),
ylabel('Average reward'),

subplot(2,1,2);
hold on
for i=1:length(EpsVals)
plot(1:NPlays,OptimalActionRatio(:,i),PlotColor{i},'LineWidth',1.5),
end
hold off
ylim([0,1])
xlim([1, NPlays])
xlabel('Plays'),
ylabel('% Optimal action'),

%Saving figure
savefilename=['N_armed_testbed'];
saveas(1,[savefilename '.eps'],'epsc');
              
disp('Finished, figure saved.');            
            
 



end



function n_armed_test()

%Inputs
epsilon=0.1; %epsilon value
alpha=0.1; %alpha step size for reward and average reward
beta=0.1; %step size for preference
PlotColor={'r', 'k', 'g'};
NPlays=1000; %number of plays for a system
NSample=2000; %number of independent systems, for statistics
%Parameters
NArm=10; %number of options (arms)
BiasMean=0.0; %the mean of the distribution from which the biases are drawn
BiasSigma=1.0;%the std. deviation of the distribution from which the biases are drawn
ArmSigma=1.0;%the std. deviation of the random reward from the arms
InitReward=0.0; %Initial reward estimate, high number to encourage exploration

%Statistics
TotalReward=zeros(NPlays,3);
OptimalActionNum=zeros(NPlays,3); %number of times the optimal action was used

for i=1:3 %for models
    %Main loop
    for j=1:NSample %for each system
    %Init
    Bias=random('Normal',BiasMean,BiasSigma,1,NArm); %biases for each arm
    [~,BestArmIndex]=max(Bias); %find optimal index
    preference=zeros(1,NArm);
    rewards=zeros(1,NArm)+InitReward; %average reward from an "arm"
    ref_reward=0;
    nplayed=zeros(1,NArm); %number of times an "arm" has been played
        for k=1:NPlays %for each play
            switch i
                case 1
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Epsilon greedy
                [rewards, nplayed, choice, currentreward]=epsilon_greedy_step(epsilon,rewards,nplayed,Bias,ArmSigma);
                case 2
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Epsilon greedy with finite step
                [rewards, choice, currentreward]=epsilon_greedy_alpha_step(epsilon,alpha,rewards,Bias,ArmSigma);
                case 3
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Reinforcement step
                [preference, ref_reward, choice, currentreward]=alpha_reinforcement_step(alpha,beta,preference,ref_reward,Bias,ArmSigma);
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Update statistics
            TotalReward(k,i)=TotalReward(k,i)+currentreward; %update total
            if choice==BestArmIndex
                OptimalActionNum(k,i)=OptimalActionNum(k,i)+1; %was this the optimal choice
            end     
        end
    end
end

OptimalActionRatio=zeros(size(OptimalActionNum));
AverageReward=zeros(size(TotalReward));
for i=1:length(OptimalActionRatio(1,:))
    OptimalActionRatio(:,i)=OptimalActionNum(:,i)./NSample; %ratio of optimal choices
    AverageReward(:,i)=TotalReward(:,i)./NSample; %the average reward per play
end


%%%%%%%
% Plots

titletext=['N armed testbed results for \epsilon-greedy, action value and reinforced learning models'];
close all;
figure(1),
set(1,'name',titletext,'units','normalized','position',[0.4,0.25,0.5,0.5])
title(titletext);

hold on
for i=1:length(OptimalActionRatio(1,:))
plot(1:NPlays,OptimalActionRatio(:,i),PlotColor{i},'LineWidth',1.5),
end
hold off
ylim([0,1])
xlim([1, NPlays])
xlabel('Plays'),
ylabel('% Optimal action'),

%Saving figure
savefilename=['N_armed_testbed_models'];
saveas(1,[savefilename '.eps'],'epsc');
              
disp('Finished, figure saved.');   




end



function [rewards, nplayed, choice, currentreward]=epsilon_greedy_step(epsilon,rewards,nplayed,Bias,ArmSigma)

if(rand()>epsilon) %deterministic case
    [~,choice]=max(rewards); %find best guess
else %stochastic case
    choice=randi(length(rewards));
end
currentreward=Bias(choice)+randn()*ArmSigma;%calculate reward for choice
%Update estimates
if nplayed(choice)>=2
    rewards(choice)=rewards(choice)+(currentreward-rewards(choice))/nplayed(choice);
else
    rewards(choice)=currentreward;
end
nplayed(choice)=nplayed(choice)+1; %chosen +1 times

end

function [rewards, choice, currentreward]=epsilon_greedy_alpha_step(epsilon,alpha,rewards,Bias,ArmSigma)

if(rand()>epsilon) %deterministic case
    [~,choice]=max(rewards); %find best guess
else %stochastic case
    choice=randi(length(rewards));
end
currentreward=Bias(choice)+randn()*ArmSigma;%calculate reward for choice
%Update estimates
rewards(choice)=rewards(choice)+(currentreward-rewards(choice))*alpha;


end


function [preference, ref_reward, choice, currentreward]=alpha_reinforcement_step(alpha,beta,preference,ref_reward,Bias,ArmSigma)

probabilities=softmax(preference); %get probabilities from preferences
[choice]=random_choice(probabilities); %choose
currentreward=Bias(choice)+randn()*ArmSigma;%calculate reward for choice
%Update estimates
preference(choice)=preference(choice)+beta*(currentreward-ref_reward); %updated preference
ref_reward=ref_reward+alpha*(currentreward-ref_reward); %updated ref reward

end

function probabilities=softmax(preference) %softmax probability function
probabilities=exp(preference)./sum(exp(preference));
end

function [choice]=random_choice(probabilities)
cumulative_dist=cumsum(probabilities);
choice=find(cumulative_dist>rand(),1,'first');
end


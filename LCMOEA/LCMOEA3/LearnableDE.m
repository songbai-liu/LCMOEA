function Offspring = LearnableDE(Problem, Population, V, k)
% The learnable differential evolution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [CR, F,proM,disM] = deal(1.0, 0.5,1,20);
    Lower = Problem.lower;
	Upper = Problem.upper;

    N = length(Population);
    CV = sum(max(0,Population.cons),2); 
    rf = sum(CV<1e-6)/length(Population);

    if rf < 0.5
        [~,rank] = sort(CV);
        winner = Population(rank(1:N/2));
        losser = Population(rank((N/2+1):N));
        rowrank1 = randperm(length(winner));
        rowrank2 = randperm(length(losser));
        winner = winner(rowrank1);
        losser = losser(rowrank2); 
    else
        [INDEX,DIS] = Association(Population,V,k);
        for i = 1:Problem.N
            if CV(INDEX(1,i)) < CV(INDEX(2,i))
                winner(i) = Population(INDEX(1,i));
                losser(i) = Population(INDEX(2,i));
            elseif DIS(INDEX(1,i),i) < DIS(INDEX(2,i),i)
                winner(i) = Population(INDEX(1,i));
                losser(i) = Population(INDEX(2,i));            
            else
                winner(i) = Population(INDEX(2,i));            
                losser(i) = Population(INDEX(1,i));            
            end
        end
    end
    
    mlp = ModelLearning(Problem, losser, winner); 


    FrontNo = NDSort(Population.objs,Population.cons,1);   
    index1  = find(FrontNo==1);
    r       = floor(rand*length(index1))+1;
    best    = index1(r);

    %% Learnable Evolutionary Search for Reproduction
    % For each solution in the current population
    if rf < 0.5
        for i = 1 : Problem.N/2
            % Choose two different random parents
		    p = randperm(Problem.N/2, 2); 
		    while p(1)==i || p(2)==i
			    p = randperm(Problem.N/2, 2); 
            end	
            % Generate an child
            Parent1 = winner(i).decs;
            Parent2 = losser(i).decs;
		    Parent3 = winner(p(1)).decs;
		    Parent4 = winner(p(2)).decs;
            Parent5 = losser(p(1)).decs;
		    Parent6 = losser(p(2)).decs;
            [~, D] = size(Parent1);
            child1 = Parent1;
            child2 = Parent2;
            [GDV, ~] = mlp.forward(child1);
            GDV = GDV.*repmat(Upper-Lower,size(GDV,1),1) + repmat(Lower,size(GDV,1),1);

            Site = rand(1,D) < CR; %CR = 1.0
            child1(Site) = child1(Site) + F*(GDV(Site)-Parent1(Site)) + F*(Parent3(Site)-Parent4(Site));
            child2(Site) = child2(Site) + F*(Parent1(Site)-Parent2(Site)) + F*(Parent3(Site)-Parent4(Site));

            %% Polynomial mutation
		    Site  = rand(1,D) < proM/D;
		    mu    = rand(1,D);
		    temp  = Site & mu<=0.5;
		    child1       = min(max(child1,Lower),Upper);
		    child1(temp) = child1(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
				    (1-(child1(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
		    temp = Site & mu>0.5; 
		    child1(temp) = child1(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
				    (1-(Upper(temp)-child1(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
		
	        %Evaluation of the new child
		    child1 = Problem.Evaluation(child1);

            child2       = min(max(child2,Lower),Upper);
		    child2(temp) = child2(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
				    (1-(child2(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
		    temp = Site & mu>0.5; 
		    child2(temp) = child2(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
				    (1-(Upper(temp)-child2(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
            child2 = Problem.Evaluation(child2);
		
		    %add the new child to the offspring population
		    Offspring(i) = child1; 
            Offspring(i+Problem.N/2) = child2; 
        end
    else
        for i = 1 : Problem.N
            % Choose two different random parents
		    p = randperm(Problem.N, 2); 
		    while p(1)==i || p(2)==i
			    p = randperm(Problem.N, 2); 
            end	
            % Generate an child
            Parent1 = Population(i).decs;
		    Parent2 = Population(p(1)).decs;
		    Parent3 = Population(p(2)).decs;
            [~, D] = size(Parent1);
            child = Parent1;
            [GDV, ~] = mlp.forward(child);
            GDV = GDV.*repmat(Upper-Lower,size(GDV,1),1) + repmat(Lower,size(GDV,1),1);
            Site = rand(1,D) < CR; %CR = 1.0
            child(Site) = child(Site) + F*(GDV(Site)-Parent1(Site)) + F*(Parent2(Site)-Parent3(Site));

            %% Polynomial mutation
		    Site  = rand(1,D) < proM/D;
		    mu    = rand(1,D);
		    temp  = Site & mu<=0.5;
		    child       = min(max(child,Lower),Upper);
		    child(temp) = child(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
				    (1-(child(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
		    temp = Site & mu>0.5; 
		    child(temp) = child(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
				    (1-(Upper(temp)-child(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
		
	        %Evaluation of the new child
		    child = Problem.Evaluation(child);
		
		    %add the new child to the offspring population
		    Offspring(i) = child; 
        end
    end
    
end

function [INDEX,DIS] = Association(Population,V,k)
    % Normalization 
    N = length(Population);
    zmin = min(Population.objs,[],1);
    zmax = max(Population.objs,[],1);
    PopObj    = (Population.objs - repmat(zmin,N,1))./(repmat(zmax-zmin,N,1));
    % Associate k candidate solutions to each reference vector
    normP  = sqrt(sum(PopObj.^2,2));
    Cosine = 1 - pdist2(PopObj,V,'cosine');
    d1     = repmat(normP,1,size(V,1)).*Cosine;
    d2     = repmat(normP,1,size(V,1)).*sqrt(1-Cosine.^2);
    DIS    = d1 + 0.25*d2;
    [~,index] = sort(d2,1);
    INDEX     = index(1:min(k,length(index)),:);
end
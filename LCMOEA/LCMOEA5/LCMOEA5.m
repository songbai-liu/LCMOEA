classdef LCMOEA5 < ALGORITHM
% <multi> <real/integer> <constrained>
% Learnable constrained multiobjective differential evolution

%------------------------------- Reference --------------------------------
% 
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter settings
            k = Algorithm.ParameterSet(2);
            Population     = Problem.Initialization();
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
            [z,znad]      = deal(min(Population.objs),max(Population.objs));
            CV = sum(max(0,Population.cons),2);
            Archive       = Population(CV == 0);
            Zmin       = min(Population.objs,[],1);
            %% Optimization
            while Algorithm.NotTerminated(Archive) 
                a          = Problem.FE/Problem.maxFE;
                if rand > 0.0
                    %Offspring  = DEgenerator2(Problem,Population);
                    if length(Archive) == Problem.N
                        %Offspring  = LearnableDE2(Problem, Archive, V, k, a);
                        %Offspring  = DEgenerator2(Problem,Archive);
                        Nt = floor((1-a)*Problem.N);
                        MatingPool = [Population(randsample(Problem.N,Nt)),Archive(randsample(Problem.N,Problem.N-Nt))];
                        [Mate1,Mate2,Mate3] = Neighbor_Pairing_Strategy(MatingPool,Zmin);
                        Offspring = OperatorDE(Problem,Mate1,Mate2,Mate3);
                    else
                        %Offspring  = DEgenerator2(Problem,Population);
                        Offspring  = LearnableDE2(Problem, Population, V, k, a);
                    end
                else
                    if a > 0.5 && length(Archive) == Problem.N
                        [FrontNo,~] = NDSort(Archive.objs,Problem.N);
                        CrowdDis = CrowdingDistance(Archive.objs,FrontNo);
                        MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                        Offspring  = OperatorGA(Problem,Archive(MatingPool));
                    else
                        MatingPool = TournamentSelection(2,Problem.N,sum(max(0,Population.cons),2));
                        Offspring  = OperatorGA(Problem,Population(MatingPool));
                    end
                end
                cv_offspring = sum(max(0,Offspring.cons),2);
                feasible_Offspring = Offspring(cv_offspring == 0);
                Zmin = min([Zmin;Offspring.objs],[],1);
                %
                if a < 0.25
                    Population = EnvironmentalSelection_TDEA([Population,Offspring],V,Problem.N,z,znad);
                    %Population = EnvironmentalSelection_Clustering2([Population,Offspring],Problem.N,z,znad,Problem.N,a);
                    %Population = Clustering([Population,Offspring],Problem.N,z,znad,Problem.N,a);
                else
                    Population = EnvironmentalSelection_Clustering2([Population,Offspring],Problem.N,z,znad,Problem.N,a);
                end
                if length(Archive) + length(feasible_Offspring) <= Problem.N
                    Archive = [Archive,feasible_Offspring];
                else
                    Archive = EnvironmentalSelection_Clustering3([Archive,feasible_Offspring],Problem.N,z,znad,Problem.N,a);
                end
            end
        end
    end
end
%generate neuropopulation class instance
close all
clc
NP=NeuroPopulation('Ike20120905_processed_original.mat', true);
verbose=true;
%%              Basics 


%% # PSTH 
%creates the average from neuron 20, 500 ms before and 500 ms after onset,
%with 20 ms bins
neuronNumber=20; timeBeforeOnset=500; timeAfterOnset=500; timeBetweenBins=20;
[~,~]=NP.createPSTH(neuronNumber, timeBeforeOnset, timeAfterOnset, timeBetweenBins, verbose);
%% #CosineTuningCurves
%if true, the figure is plotted
% NP.createCosineTuningCurve(20, true);
%the parameter shows whether they are scaled (true) or not (false)
isScaled=true;
NP.plotPreferredDirections(isScaled);
%%              Dimensionality Reduction


%% PCA
%verbose (true) plots points in 3D space and outputs the explained variance
%of 3, 10 and 20 principal components
dimensions=3;
% NP.findPrincipleComponents(verbose)

%% Probabilistic PCA
% NP.findProbabilisticPrincipalComponents(dimensions,verbose);
%% Factor Analysis
% NP.applyFactorAnalysis(dimensions,verbose);
%%              Decoders


%% Linear Regression

%% Population Vector Algorithm

%% Optimal Linear Estimator

%% Kalman Filter

%%              Classification

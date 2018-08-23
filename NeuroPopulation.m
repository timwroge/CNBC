%%  NeuroPopulation class
%   This class is supposed to take neural data 
%   in the form of the struct 
%   and perform operations on the data for dimensionality reduction, neural
%   data decoding, and classification. Documentation for each function ,
%   unless obvious from name should be available under the function headers

%% Timothy Wroge
%  July 26, 2018

%%                          MIT License

% Permission is hereby granted, free of charge, to any person 
% obtaining a copy of this software and associated documentation
% files (the "Software"), to deal in the Software without restriction,
% including without limitation the rights to use, copy, modify, merge,
% publish, distribute, sublicense, and/or sell copies of the Software, 
% and to permit persons to whom the Software is furnished to do so, subject
% to the following conditions:
% 
% The above copyright notice and this permission notice shall be 
% included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
% OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
% DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
% OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
% OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

classdef NeuroPopulation < handle

    properties (SetAccess =private)
        NeuralData;
        MetaData;
        Neurons;
        timeMovementOnset;
        timeMovementEnd;
        Directions;
        numberNeurons;
        anglesAndSpikeCounts;
        firingRateCovariance;
        firingRateMean;
        zeroCenteredObservations;
        Marker;
        spikeCounts;
        
    end
    methods (Static)
        function solution=atand(x,y)
            if x<0
                if y< 0
                    solution=atand(y/x)+180;
                else
                    solution=atand(y/x)-180;
                end
            else
                solution=atand(y/x);
            end

        end
        function plotEmbeddings(embeddingsAndAngle, plotTitle)
                figure

                angles=[0, 45, 90, 135, 180,225,270,315];

                colorMapping=jet(length(angles));

                for iii =1:length(angles)
                    currentAngle=angles(iii);
                    indexes=embeddingsAndAngle(:, 1)==currentAngle;
                    embeddingsForThisAngle=embeddingsAndAngle(indexes, :);
                    hold on,plot3(embeddingsForThisAngle(:,2),embeddingsForThisAngle(:, 3),...
                        embeddingsForThisAngle(:,4),  'Color', colorMapping(iii, :),'Marker', '.', ...
                        'linestyle', 'none');
                end
                xlabel('Z-1');
                ylabel('Z-2');
                zlabel('Z-3');
                title(plotTitle);
        end
    end
    methods
        function self = NeuroPopulation(NeuralDataFilename, dataIsInWorkspace)
            %The constructor of the class instance
            %this will load the data from the filename
            if (dataIsInWorkspace)
                self.NeuralData=evalin('base', 'Data');
                self.Marker=Marker('',dataIsInWorkspace);
                ov = [self.NeuralData.Overview];
                success = [ov.trialStatus] == 1;
                self.NeuralData = self.NeuralData(success);
            else
                self.NeuralData= load(NeuralDataFilename);
                self.NeuralData=self.NeuralData.Data;
                self.Marker=Marker(NeuralDataFilename,dataIsInWorkspace);
                ov = [self.NeuralData.Overview];
                success = [ov.trialStatus] == 1;
                self.NeuralData = self.NeuralData(success);
            end
            
            self.MetaData=[self.NeuralData.Overview];
            %load the time movement onset and spike data into a separate
            %array
            self.Directions=[];
            self.timeMovementOnset=[];
            self.timeMovementEnd=[];
            self.Neurons={};
            %find number of trials
            
            len=size(self.NeuralData,2);
            %the last trial is null
            for ii=1:len-1
                self.Directions=[self.Directions, self.NeuralData(ii)...
                    .TrialData.reachAngle];
                if(isempty(self.timeMovementOnset))
                    self.timeMovementOnset=[self.NeuralData(ii)...
                        .TrialData.timeMoveOnset];
                else
                    self.timeMovementOnset=[self.timeMovementOnset;...
                        self.NeuralData(ii).TrialData.timeMoveOnset];

                end
                if(isempty(self.timeMovementEnd))
                    self.timeMovementEnd=[self.NeuralData(ii)...
                        .TrialData.timeMoveEnd];
                   
                else
                    self.timeMovementEnd=[self.timeMovementEnd;...
                        self.NeuralData(ii).TrialData.timeMoveEnd];

                end
                if(isempty(self.Neurons))
                    allTimeStamps={self.NeuralData(ii).TrialData...
                        .spikes.timestamps};
                    numberOfNeurons=max(size(allTimeStamps));
                    for hh=1:numberOfNeurons
                    %make 
                    self.Neurons{ii, hh}=allTimeStamps{hh};
                    end
                else
                    allTimeStamps={self.NeuralData(ii).TrialData...
                        .spikes.timestamps};
                    numberOfNeurons=max(size(allTimeStamps));
                    for hh=1:numberOfNeurons
                    %make 
                    self.Neurons{ii, hh}=allTimeStamps{hh};
                    end
                end
            end
            
            self.anglesAndSpikeCounts=pmdDataSetup(self.NeuralData);
            self.numberNeurons=size(self.anglesAndSpikeCounts, 2)-1;
            
            %Find the covariance and mean of the spikes in order to find
            %the principal components
            spikes=(self.anglesAndSpikeCounts(:,2:end));
            self.spikeCounts=spikes;
            self.firingRateCovariance=cov(spikes);
            %take mean along columns (neurons)
            self.firingRateMean=mean(spikes,1);
            self.zeroCenteredObservations=zeros(size(spikes));
            for jj=1:size(spikes, 1)
                self.zeroCenteredObservations(jj, :)=spikes(jj,:)-self.firingRateMean;
            end

            %go through the neuron structure and find any nonfiring neurons
            %create a variable sum that is the total value of all the
            %spikecounts
            indexesToDelete=[];
            for ii=1:numberOfNeurons
                spikeSum=0;
                for jj=1:len-1
                    spikesForTrial=self.Neurons{jj,ii};
                    spikeSum=spikeSum+sum(spikesForTrial);
                end
                if spikeSum==0
                    %delete row of neurons
                    indexesToDelete=[indexesToDelete,ii];
                end
            end
            indexesToDelete=sort(indexesToDelete,'descend');
            for kk=1:length(indexesToDelete)
                self.Neurons(:,indexesToDelete(kk))=[];
            end
        end

        
        
        %now load the preferred angle directions and spikes at that angle
        
        %%%%%%%%%%%%%DATA STRUCTURES%%%%%%%%%%%%%
        %*Neurons* will be a cell array where each cell is the spike
        %timestamps for each trial (index)
        
        %*timeMovementOnset* is a normal array with the time movement onset
        %for each trial (index)

        %*anglesAndSpikeCounts* is an array for easier analysis for the
        %cosine tuning curves
        % rows = trials
        % column 1 = target angle, columns 2:N+1 = firing rate for all N neurons
        %% Basic Regressions and Plots
        function [sortedSpikes, sortedSpikesWithAngles]=createPSTH(self, neuronNumber, timeBefore, timeAfter, timeBetweenBins, verbose)
            %takes neural data and creates Peristimulus Time Histogram
            %find number of trials
            trials=length(self.Neurons);
            
            %preallocate memory
            spikeData{trials}=0;
            for ii=1:trials
                spikeData{ii}=self.Neurons{ii,neuronNumber};
            end
            %the time movement onset for these neurons is the same as the
            %timeMovementOnset parameter
            
            %concatinating all the spikes in the interval above will
            %and passing to the histogram function will create the average
            %take the window to be [-timeBefore in ms to timeAfter in ms] after onset. This
            % should sort the all the spikes into the useful data
            
            sortedSpikes=[];
            beforeOnset=self.timeMovementOnset-timeBefore;
            afterOnset=self.timeMovementOnset+timeAfter;
            angles=[0, 45, 90, 135, 180,225,270,315];
            sortedSpikesWithAngles{8}=0;
            %take the sample histogram for a single trial and add up along
            %the major dimension of activations to see the average plot 
            for ii=1:trials
                if not(isempty(spikeData{ii}))
                    numberOfSpikes=max(size(spikeData{ii}));
                    for jj=1:numberOfSpikes
                        spikes=spikeData{ii};
                        %check if spike is within the range
                        if spikes(jj) >= beforeOnset(ii) && spikes(jj)...
                                <= afterOnset(ii)
                            %this adds all the data to the array of spikes
                            tempSpikes = double(spikes(jj))...
                                -self.timeMovementOnset(ii);
                            sortedSpikes=[sortedSpikes,tempSpikes];
                            %this sorts the data into the directions as
                            %well
                            for kk=1:8
                                if self.Directions(ii) == angles(kk)
                                    %append the data to this angle array
                                    sortedSpikesWithAngles{kk}=[tempSpikes...
                                        ,sortedSpikesWithAngles{kk}];
                                    break
                                end
                                
                            end
                            
                        end
                    end
                end
            end
            
            %plot the histogram
            %the frequency of the spike rate is related to the number of
            %firing in a certain amount of time, (0.02s)
            if (verbose)
                histogram((sortedSpikes), 'binedges', -timeBefore:timeBetweenBins:timeAfter);
                order=[4,3,2,5,0, 1, 6, 7, 8];
                title(['Peristimulus Time Histogram for Neuron ', num2str(neuronNumber)])
                xlabel('Time (ms)')
                ylabel('Activation (Hz)');

                figure;

                for iii=1:9
                    if order(iii) ~= 0
                        subplot(3, 3, iii)
                        histogram(sortedSpikesWithAngles{order(iii)}, 'binedges', -timeBefore:timeBetweenBins:timeAfter);
                        title(angles(order(iii)) );
                    end
                end
            end
            %title('Spike angles and stuff')
        end
        function [baseline,modulationDepth,preferredDirection ]=createCosineTuningCurve(self, neuronNumber, draw)
            function theta_pd= getPreferredDirection(bx, by)
                theta_pd=NeuroPopulation.atand(bx, by);
            end

            neuronIndex=neuronNumber+1;
            anglesAndSpikes=self.anglesAndSpikeCounts;
            %the equation to solve for is f_r = b_0+b_x+b_y (sin(theta_m))
            %and, b_x=b_m cos(theta_p,d)
            %and, b_y=b_m sin(theta_p,d)
            
            %unknowns are b_m, b_0 and theta_pd
            cosAngles=cosd(anglesAndSpikes(:, 1));
            sinAngles=sind(anglesAndSpikes(:, 1));
            firingRates=anglesAndSpikes(:, neuronIndex);
            
            trials=size(anglesAndSpikes,1);
            %the first column is all ones because baselines is a constant
            %offset
            X=[ones(trials, 1), cosAngles, sinAngles];
            
            Y=firingRates;
            %Run Regression on coefficients 
            coefficients=(X' *X)\X'*Y;
            
            baseline=coefficients(1);
            modulationDepth=sqrt(coefficients(2)^2+coefficients(3)^2);
            preferredDirection=getPreferredDirection(coefficients(2),coefficients(3));
            
            %Plot average firing rate per target
            if (draw)
                figure
                plot(anglesAndSpikes(:, 1),firingRates, '.');

                %now plot the cosine tuning curve

                x=0:.1:350;
                y=baseline+modulationDepth*cosd(preferredDirection-x);
                
                hold on, plot(x, y, '.');   
            end
        end
        %% Dimensionality Reduction
        function plotPreferredDirections(self, scale)
            %the number of neurons is the same as the number of columns
            %minus 1 (first column is the angle)
            numberOfNeurons=size(self.anglesAndSpikeCounts, 2)-1;
            neuralFittedData=zeros(numberOfNeurons, 3);
            
            for ii=1:numberOfNeurons
                [baseline,modulationDepth,preferredDirection ]=self.createCosineTuningCurve(ii, false);
                
                neuralFittedData(ii, 1)=baseline;
                neuralFittedData(ii, 2)=modulationDepth;
                neuralFittedData(ii, 3)=preferredDirection;
            end
            directions=neuralFittedData(:,3);
            modulationDepths=neuralFittedData(:,2);
            
               
            figure
            coordinates=zeros(numberOfNeurons, 2);
            %plot at first the largest modulation depth
            
            for ii=1:numberOfNeurons
                x=cosd(directions(ii));
                y=sind(directions(ii));
                
                if scale
                    x=x*modulationDepths(ii);
                    y=y*modulationDepths(ii);
                end
                coordinates(ii, 1)=x;
                coordinates(ii, 2)=y;
                
                if ii ~= 1
                    hold on,compass(x,y)
                else 
                    compass(x,y)
                    if scale
                        % autoscale the plot to the max of the modulation
                        % depth
                        hiddenArrow=compass(max(neuralFittedData(:,2)), 0);
                        hiddenArrow.Color = 'none';
                    end
                end
            end
            
        end
        function findPrincipleComponents(self, verbose)
            function embeddings=projectIntoPrincipleComponentSpace(PricipleComponents, dimensions, points)
                embeddings=[];
                spikes=points;
                mean=[];
                
                for trial=1:size(spikes, 1)
                    mean=[mean;self.firingRateMean];
                end
                for ii=1:dimensions
                    PCScores=(spikes-mean)*PricipleComponents(:,ii);
                    embeddings=[embeddings,PCScores];
                end
            end
            
            [PricipleComponents,EigenvalueMatrix]=eig(self.firingRateCovariance);
            %this sh
            len=size(EigenvalueMatrix,1);
            %preallocate memory for eigenvalues
            Eigenvalues=sum(EigenvalueMatrix);
            %make eigenvalues in correct order (max, ...., min)
            Eigenvalues=fliplr(Eigenvalues);
            %reorient principle components to stay in place
            PricipleComponents=fliplr(PricipleComponents);
            if verbose
                figure
                number=1:len;
                plot(number,Eigenvalues);
                xlabel('Eigenvalue rank');
                ylabel('Variance');
                title('Eigenvalues of Covariance Matrix');
                totalVariance=sum(Eigenvalues);
                for kk=[3,10,20]
                    explainedVariance=sum(Eigenvalues(1:kk))/totalVariance;
                    disp(['The explained variance for the first ' , num2str(kk),' eigenvectors is: ', num2str(explainedVariance)])
                end
                dimensions=3;
                %get the first 3 principle components
                embeddings=projectIntoPrincipleComponentSpace(PricipleComponents, dimensions, self.anglesAndSpikeCounts(:, 2:end));
                embeddingsAndAngle=[self.anglesAndSpikeCounts(:,1), embeddings];
                 
                %plot the data for each reach individually with its
                %associated color, make a vector with the length 8,3 with
                %the associated color for each reach using jet
                %create color jet
                NeuroPopulation.plotEmbeddings(embeddingsAndAngle, 'PCA of Neural Population Data');
            end
        end
        function applyFactorAnalysis(self,dimensions, verbose)
            function [ExpectedZ]=Expectation( W, Psi)
                C=W*W'+Psi;
                ExpectedZ=W'*C^(-1)*centeredObservations;
%                 logLikelihood=-numberTrials*neuralDimensions/2 *log(2*pi*;
            end
            function [newW, newPsi]=Maximization(ExpectedZ, W, Psi)
                C=W*W'+Psi;
                covZ=eye(latentDimensions)-W'*(inv(C))*W;
                newW=centeredObservations*ExpectedZ'*(numberTrials*covZ+ExpectedZ*ExpectedZ')^(-1);
                newPsi=1/numberTrials*(diag(diag(centeredObservations*centeredObservations'-newW*ExpectedZ*centeredObservations')));
            end
            function embeddings=projectIntoFactorSpace()
                C=W*W'+Psi;
                embeddings=W'*C^(-1)*centeredObservations;
            end
            centeredObservations=self.zeroCenteredObservations;
            centeredObservations=centeredObservations';
            numberTrials=size(centeredObservations,2);
            %initialize weight matrix W and Psi
            latentDimensions=dimensions;
            neuralDimensions=self.numberNeurons;
            W=ones(neuralDimensions,latentDimensions);
            Psi=diag(diag(ones(neuralDimensions)));
            %perform expectation , maximization algorithm for 50 iterations
            for count=1:50
                [ExpectedZ]=Expectation( W, Psi);
                [W, Psi]=Maximization(ExpectedZ, W, Psi);
            end
            if(verbose)
                embeddings=projectIntoFactorSpace();
                embeddingsAndAngle=[self.anglesAndSpikeCounts(:,1), embeddings'];
                NeuroPopulation.plotEmbeddings(embeddingsAndAngle, 'Factor Analysis of Neuro Population Data');
            end
            
        end
        function findProbabilisticPrincipalComponents(self,dimensions, verbose)
            function [ExpectedZ]=Expectation( W, Variance)
                C=W*W'+Variance*eye(neuralDimensions);
                ExpectedZ=W'*C^(-1)*centeredObservations;
            end
            function [newW, newVariance]=Maximization(ExpectedZ, W, Variance)
                C=W*W'+Variance*eye(neuralDimensions);                
                covZ=eye(latentDimensions)-W'*(inv(C))*W;
                newW=centeredObservations*ExpectedZ'*(numberTrials*covZ+ExpectedZ*ExpectedZ')^(-1);
                newVariance=1/numberTrials/neuralDimensions*(trace((centeredObservations*centeredObservations'-newW*ExpectedZ*centeredObservations')));;
            end
            function embeddings=projectIntoFactorSpace()
                C=W*W'+Variance*eye(neuralDimensions); %C=1/2*C*(C');
                embeddings=W'*C^(-1)*centeredObservations;
            end
            centeredObservations=self.zeroCenteredObservations;
            centeredObservations=centeredObservations';
            numberTrials=size(centeredObservations,2);
            %initialize weight matrix W and Variance
            latentDimensions=dimensions;
            neuralDimensions=self.numberNeurons;
            W=ones(neuralDimensions,latentDimensions);
            Variance=1;
            %perform expectation , maximization algorithm for 50 iterations
            for count=1:50
                [ExpectedZ]=Expectation( W, Variance);
                [W, Variance]=Maximization(ExpectedZ, W, Variance);
            end
            if(verbose)
%                 fprintf('[+] EM Algorithm converged at iteration %i with a log-likelihood of %d \n', count, newLogLikelihood);
                embeddings=projectIntoFactorSpace();
                embeddingsAndAngle=[self.anglesAndSpikeCounts(:,1), embeddings'];
                NeuroPopulation.plotEmbeddings(embeddingsAndAngle, 'Probabilisitic PCA of Neuro Population Data');
            end
        end
        %% Decoding
        %first, calculate position velocity, and spike counts in 50 ms bins
        % 100 ms before movement onset til the end of the movement (0 ms)
        
        %the pmdDataSetup will remove all non-firing neurons
        
        %this data will be used for the decoders below
        
        %the data for the position and velocity of the marker will be
        %available in the self.Marker object, the specific movement
        %directions and sampling of the data to be sorted is done in the
        %init method (NeuroPopulation function)
        
        function linearRegressionAngleEstimation(self)
            %try to regress the velocity as a function of firing rate
            % velocity = a * firing_rate 
            % angle= Neuropopuluation.atan(velocity.x, velocity.y);
            
            %sort data for angle estimation
            %the targets for the linear estimation will be the velocity
            %the error is the inverse tanget of the different velocities
            
            %in this case, we can just use pmd Data setup for the training
            %and testing data
            
            trainingPercentage=80;
            angles=self.anglesAndSpikeCounts(:,1);
            firingRates=self.anglesAndSpikeCounts(:,2:end);
            numberOfNeurons=size(firingRates,2);
            trials=size(firingRates,1);
            %add the column of ones to add a bias for the data (so the
            %regression does not have a y intercept of 0)
            X=[ones(trials,1),firingRates];
            y=angles;
            indexOfTraining=floor(80/100*trials);
            X_train=X(1:indexOfTraining,:);
            X_test=X((indexOfTraining+1):end,:);
            y_train=y(1:indexOfTraining,:);
            y_test=y((indexOfTraining+1):end,:);
            
            %regress training data
            A=(X_train'*X_train)^(-1) *X_train'*y_train;
            %see how it did
            %absolute angular error
            y_hat=X_test*A;
            absoluteAngularError=abs(y_hat-y_test);
            %make a histogram of the absolute angular error
            figure;
            histogram(absoluteAngularError);
            title('Linear Regression Absolute Angular Error');
            xlabel('Angular Error');
            
        end
        function linearRegressionMovementEstimation(self)
            %get X and Y from getDecodingData
            [X,y]=self.getDecodingData();
            indexOfTraining=floor(80/100*trials);
            X_train=X(1:indexOfTraining,:);
            X_test=X((indexOfTraining+1):end,:);
            y_train=y(1:indexOfTraining,:);
            y_test=y((indexOfTraining+1):end,:);
            
            %regress training data
            A=(X_train'*X_train)^(-1) *X_train'*y_train;
            %see how it did
            %absolute angular error
            y_hat=X_test*A;
            absoluteAngularError=abs(y_hat-y_test);
            %make a histogram of the absolute angular error
            figure;
            histogram(absoluteAngularError);
            title('Linear Regression Movement Estimation');
            xlabel('Angular Error');
        end
        function populationVectorAlgorithmForAngleAndMovementEstimation(self)
            
        end
        function OLEMovementTrajectory(self)
            %this is the optimal linear estimator for use in decoding the
            %movement trajectories
        end
        function kalmanFilterAngleAndMovementEstimation(self)
           %estimate movement trajectory
           %the goal of the kalman filter is to take the incoming neural
           %data signal and decode it to the movement velocities and
           %positions in real time. In order to accomplish this, we adopt a
           %supervised approach which tries to map the observation X from
           %the internal state Z which is the intended movement of the
           %monkey (velocities and stuff)
           
        end
        
        function [X,y]=getDecodingData(self)
            function spikeCounts=getSpikeCounts(trial, time)
                % calculate spike counts in 50 ms bins
                % 100 ms before movement onset til the end of the movement
                
                %first get 
                %movement onset and resample so it is to the nearest 50th
                firstTime=time(1);firstTime=round(firstTime/50)*50;
                %movement end and resample so it is to the nearest 50th
                secondTime=time(2);secondTime=round(secondTime/50)*50;
                binEdges=firstTime:50:secondTime;
                %next, grab the spike times 
                spikeTimes={self.Neurons{trial,:}};
                %make data that will be populated later
                rows=max(size(binEdges))-1;
                columns=size(spikeTimes,2);
                % number of neurons
                spikeCounts=zeros(rows, columns);
                %now use histcounts to get the spikeCounts using the bin
                %edges
                for ii=1:columns
                    %get the spike counts for the ii th neuron
                    tempCounts=histcounts(spikeTimes{ii}, 'binedges', binEdges);
                    spikeCounts(:, ii)=tempCounts;
                end
%                 disp(['Spike Counts: ',spikeCounts]);
                %the way the data should look
                   %      neuron 1  neuron 2   neuron 3 ... neuron N
                   %      spikes0-50 ...                ... spikes 0
                   %      spikes50-100 ...               ...spikes 50-100
            end
            %get positions, velocities and times downsampled to 50 ms
            
            %time will be 100 ms before movement onset (index 1) until
            %movement end (index 2)
            %continuously add rows based on data obtained from spikecounts
            %and the postion, velocity and time
            % decoding target is [position.x; position.y;velocity.x;
            % velocity.y] 
            
            %find number of trials
            trials=self.Marker.trials;
            y=[];
            X=[];
            for trial =1:trials
                [position,velocity, time]=self.Marker.getPositionVelocitesAndTime(trial);
                %make sure number of rows match
                if trial ~=1
                y=[y;position.x, position.y, velocity.x, velocity.y];
                X=[X;getSpikeCounts(trial, time)];
                else
                y=[position.x, position.y, velocity.x, velocity.y];
                X=getSpikeCounts(trial, time);
                end
                    if size(X,1) ~=size(y,1) 
                        X
                        y
                        fprintf('Error in trial %i: Size of x (%i, %i), does not match size of y (%i, %i)\n',trial, size(X,1), size(X,2),size(y,1),size(y,2));
                        ME= MException('MATLAB:DataSizesIncorrect','The number of dimensions for the target does not match the dimensions for the observation');
                        throw(ME)
                    end
            end
            
        end
    end
    
        
end   

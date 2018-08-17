classdef Marker <handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess =private)
        positions;
        velocity;
        times;
        MarkerData;
        timeMovementOnset;
    end
    
    methods
        function self = Marker(MarkerDataFilename, dataIsInWorkspace)
            if (dataIsInWorkspace)
                self.MarkerData=evalin('base', 'Data');
                ov = [self.MarkerData.Overview];
                success = [ov.trialStatus] == 1;
                self.MarkerData = self.MarkerData(success);
            else
                self.MarkerData=load(MarkerDataFilename);
                self.MarkerData=self.MarkerData.Data;
                ov = [self.MarkerData.Overview];
                success = [ov.trialStatus] == 1;
                self.MarkerData = self.MarkerData(success);                
            end



            trials=size(self.MarkerData, 1);
            self.timeMovementOnset=zeros(1, trials);
            self.positions.x=zeros(1, trials);
            self.positions.y=zeros(1, trials);        
            %because we are dealing with deltas, the first trial cannot be
            %used
            
            self.velocity.x=zeros(1, trials);
            self.velocity.y=zeros(1, trials);  
            %this is the sampling rate in milliseconds
            samplingRate=25/3;
            fprintf('Number of trials: %i', trials);
            for trial=1:trials
                self.timeMovementOnset(trial)=self.MarkerData(trial).TrialData.timeMoveOnset;
                self.positions.x=self.MarkerData(trial).TrialData.Marker.rawPositions(:,2);
                self.positions.y=self.MarkerData(trial).TrialData.Marker.rawPositions(:,3);
                self.times=self.MarkerData(trial).TrialData.Marker.rawPositions(:,3);
                if trial ~=1
                    disp(trial)
                    self.velocity.x=(self.positions.x(trial)-self.positions.x(trial-1))/samplingRate;
                    self.velocity.y=(self.positions.y(trial)-self.positions.y(trial-1))/samplingRate;
                end
            end
            
            

        end
        function positions= getPostions(self)
            positions=self.positions;
        end
        function velocities=getVelocities(self)
            velocities=self.velocity;
        end
        function times=getTimes(self)
            times=self.times;
        end
        
    end
    
end


classdef huberRegressionLayer < nnet.layer.RegressionLayer ...
        & nnet.layer.Acceleratable
    % custom regression layer with huber loss function

    properties
        Delta;
    end
    
    methods
        function layer = huberRegressionLayer(name,delta)

			layer.Name = name;
            layer.Delta = delta;
            layer.Description = 'Huber loss';
        end
        
        function loss = forwardLoss(layer, Y, T)

            d = layer.Delta;

            loss = huber(Y,T,'DataFormat','SBT','TransitionPoint',d);
        end
    end
end
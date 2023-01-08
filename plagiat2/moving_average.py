        
from etna.models.seasonal_ma import SeasonalMovingAverageModel

class MovingAverageModel(SeasonalMovingAverageModel):
     

        def __init__(self, window: i=5):
                self.window = window
                super().__init__(window=window, seasonality=1)
    
__all__ = ['MovingAverageModel']
    

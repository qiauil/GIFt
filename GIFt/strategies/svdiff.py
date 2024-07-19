from ..modules.svdiff import *
from . import FineTuningStrategy, merger_strategy
from ..utils import factories as fts

class SVDiffLinearFineTuningStrategy(FineTuningStrategy):
    
    def __init__(self, train_bias: bool = False) -> None:
        FineTuningStrategy.__init__(self,
                                    [
                                        (fts.c_cname_equal2("Linear"),fts.a_replace(SVDiffLinear),{"train_bias":train_bias}),
                                    ]
                                    )
        
class SVDiffConvFineTuningStrategy(FineTuningStrategy):
    
    def __init__(self,train_bias: bool = False) -> None:
        config={"train_bias":train_bias}
        FineTuningStrategy.__init__(self,
                                        [
                                            (fts.c_cname_equal2("Conv1d"),fts.a_replace(SVDiffConv1d),config),
                                            (fts.c_cname_equal2("Conv2d"),fts.a_replace(SVDiffConv2d),config),
                                            (fts.c_cname_equal2("Conv3d"),fts.a_replace(SVDiffConv3d),config),
                                        ]
                                    )

def SVDiffAllFineTuningStrategy(train_bias: bool = False):
    return merger_strategy([SVDiffLinearFineTuningStrategy(train_bias),
                           SVDiffConvFineTuningStrategy(train_bias)])